//! Validates the column defaults of a replacement schema, following Delta Kernel Java's
//! `ColumnDefaults.validateSchema` and `ColumnDefaults.validateLiteral` apart from the literal
//! forms listed on [`validate_literal`].

use std::ops::RangeInclusive;

use crate::schema::{
    DataType, DecimalType, PrimitiveType, StructType, try_collect_column_defaults,
};
use crate::table_configuration::TableConfiguration;
use crate::table_features::TableFeature;
use crate::utils::require;
use crate::{KernelError, KernelResult};

/// Rejects column defaults in `schema` that `table_config` does not allow.
///
/// Any default requires the `allowColumnDefaults` feature, and a table with that feature must have
/// `delta.enableIcebergCompatV3` enabled, even if `schema` has no defaults. Every default must be
/// a literal of its field's type, as accepted by [`validate_literal`].
pub(crate) fn validate_column_defaults(
    table_config: &TableConfiguration,
    schema: &StructType,
) -> KernelResult<()> {
    let defaults_supported = table_config.is_feature_supported(&TableFeature::AllowColumnDefaults);
    if defaults_supported {
        ensure_iceberg_compat_v3_for_column_defaults(table_config)?;
    }
    let defaults = try_collect_column_defaults(schema)?;
    if !defaults_supported {
        require!(
            defaults.is_empty(),
            KernelError::schema(
                "Found column defaults in the schema but the table does not support the \
                 allowColumnDefaults table feature"
            )
        );
        return Ok(());
    }
    for (path, default) in defaults {
        match validate_literal(default.data_type(), default.raw_sql()) {
            Ok(()) => {}
            Err(LiteralError::NotLiteral) => {
                return Err(KernelError::schema(format!(
                    "Only literal values are supported for default values. {} is an invalid \
                     default value for field '{path}' of type {}",
                    default.raw_sql(),
                    default.data_type()
                )));
            }
            Err(LiteralError::UnsupportedType) => {
                return Err(KernelError::unsupported(format!(
                    "Kernel does not support default values for data type {}: {path}",
                    default.data_type()
                )));
            }
        }
    }
    Ok(())
}

/// Rejects a table with the `allowColumnDefaults` feature whose `delta.enableIcebergCompatV3` is
/// not `true`. Kernel supports column defaults only on IcebergCompatV3 tables.
fn ensure_iceberg_compat_v3_for_column_defaults(
    table_config: &TableConfiguration,
) -> KernelResult<()> {
    require!(
        table_config.table_properties().enable_iceberg_compat_v3 == Some(true),
        KernelError::unsupported(
            "Kernel supports the allowColumnDefaults table feature only when icebergCompatV3 is \
             enabled"
        )
    );
    Ok(())
}

#[derive(Debug, PartialEq)]
enum LiteralError {
    NotLiteral,
    UnsupportedType,
}

/// Checks that `value` is a literal of `data_type`, following Java's `validateLiteral`.
///
/// String and binary literals must be enclosed in single or double quotes. Other literals may be
/// quoted. Numbers must parse and fit the type, a decimal must fit the precision and scale, a date
/// must be ISO `yyyy-MM-dd`, a timestamp ISO date-time with an offset, and a `timestamp_ntz` ISO
/// date-time without one. Like Java's `Boolean.parseBoolean`, any value is accepted for a boolean.
/// Byte and non-primitive types are not supported, and `NULL` is not a literal.
///
/// Two forms that Java accepts are rejected: a timestamp with a `[zone]` suffix, because checking
/// the zone ID needs a time-zone database that Kernel does not always include, and integer or
/// decimal digits outside ASCII, because Rust's standard library has no table of Unicode decimal
/// digits. Rejecting them rather than accepting them unchecked ensures Kernel never commits a
/// default that Java rejects.
fn validate_literal(data_type: &DataType, value: &str) -> Result<(), LiteralError> {
    let DataType::Primitive(primitive) = data_type else {
        return Err(LiteralError::UnsupportedType);
    };
    let (unquoted, quoted) = strip_quotes(value);
    let valid = match primitive {
        PrimitiveType::String | PrimitiveType::Binary => quoted,
        PrimitiveType::Long => is_java_integer::<i64>(unquoted),
        PrimitiveType::Integer => is_java_integer::<i32>(unquoted),
        PrimitiveType::Short => is_java_integer::<i16>(unquoted),
        PrimitiveType::Float | PrimitiveType::Double => is_java_floating_point(unquoted),
        PrimitiveType::Decimal(decimal) => fits_decimal(unquoted, decimal),
        PrimitiveType::Boolean => true,
        PrimitiveType::Date => is_iso_local_date(unquoted),
        PrimitiveType::Timestamp => split_offset(unquoted).is_some_and(is_iso_local_date_time),
        PrimitiveType::TimestampNtz => is_iso_local_date_time(unquoted),
        _ => return Err(LiteralError::UnsupportedType),
    };
    if valid {
        Ok(())
    } else {
        Err(LiteralError::NotLiteral)
    }
}

/// Removes enclosing single or double quotes and reports whether there were any.
fn strip_quotes(value: &str) -> (&str, bool) {
    let inner = ['\'', '"'].into_iter().find_map(|quote| {
        (value.len() > 1)
            .then(|| value.strip_prefix(quote)?.strip_suffix(quote))
            .flatten()
    });
    match inner {
        Some(inner) => (inner, true),
        None => (value, false),
    }
}

/// Java's `Long.parseLong`, `Integer.parseInt`, and `Short.parseShort`: an optional sign followed
/// by ASCII digits that fit the type. Rust's integer parsing accepts the same syntax.
fn is_java_integer<T: std::str::FromStr>(value: &str) -> bool {
    value.parse::<T>().is_ok()
}

/// Java's `Double.parseDouble`: surrounding whitespace, an optional `f`, `F`, `d`, or `D` suffix,
/// the special values `NaN` and `Infinity`, and decimal or hexadecimal notation. Out-of-range
/// values are valid because they round to zero or infinity.
fn is_java_floating_point(value: &str) -> bool {
    let value = value.trim_matches(|c: char| c <= ' ');
    let unsigned = value.strip_prefix(['+', '-']).unwrap_or(value);
    if unsigned == "Infinity" || unsigned == "NaN" {
        return true;
    }
    let unsigned_number = unsigned
        .strip_suffix(['f', 'F', 'd', 'D'])
        .unwrap_or(unsigned);
    if let Some(hex) = unsigned_number
        .strip_prefix("0x")
        .or_else(|| unsigned_number.strip_prefix("0X"))
    {
        return is_hex_significand_and_exponent(hex);
    }
    // Parse with the sign so that a second sign, as in `+-1`, is rejected.
    let number = value.strip_suffix(['f', 'F', 'd', 'D']).unwrap_or(value);
    number
        .chars()
        .all(|c| c.is_ascii_digit() || matches!(c, '+' | '-' | '.' | 'e' | 'E'))
        && number.parse::<f64>().is_ok()
}

/// Java's hexadecimal floating-point form after `0x`: hex digits with an optional `.`, then a
/// binary exponent `p` or `P` with an optional sign.
fn is_hex_significand_and_exponent(hex: &str) -> bool {
    let Some((significand, exponent)) = hex.split_once(['p', 'P']) else {
        return false;
    };
    let (integer, fraction) = significand.split_once('.').unwrap_or((significand, ""));
    let exponent = exponent.strip_prefix(['+', '-']).unwrap_or(exponent);
    !(integer.is_empty() && fraction.is_empty())
        && integer
            .bytes()
            .chain(fraction.bytes())
            .all(|b| b.is_ascii_hexdigit())
        && !exponent.is_empty()
        && exponent.bytes().all(|b| b.is_ascii_digit())
}

/// Java's `new BigDecimal(value)` followed by the check that the scale and precision fit
/// `decimal`. A negative scale (from a positive exponent) always fits.
fn fits_decimal(value: &str, decimal: &DecimalType) -> bool {
    let Some((digits, scale)) = parse_big_decimal(value) else {
        return false;
    };
    let significant = digits.trim_start_matches('0').len().max(1);
    scale <= i32::from(decimal.scale()) && significant <= usize::from(decimal.precision())
}

/// Parses `[+-]digits[.digits][(e|E)[+-]digits]` into the unscaled digits and the scale, as Java's
/// `BigDecimal(String)` constructor does.
fn parse_big_decimal(value: &str) -> Option<(String, i32)> {
    let value = value.strip_prefix(['+', '-']).unwrap_or(value);
    let (mantissa, exponent) = match value.find(['e', 'E']) {
        Some(index) => (&value[..index], Some(&value[index + 1..])),
        None => (value, None),
    };
    let (integer, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let all_digits = |s: &str| s.bytes().all(|b| b.is_ascii_digit());
    if integer.len() + fraction.len() == 0 || !all_digits(integer) || !all_digits(fraction) {
        return None;
    }
    let exponent = match exponent {
        Some(exponent) => {
            let unsigned = exponent.strip_prefix(['+', '-']).unwrap_or(exponent);
            if unsigned.is_empty() || !all_digits(unsigned) {
                return None;
            }
            exponent.parse::<i64>().ok()?
        }
        None => 0,
    };
    // Like Java's `BigDecimal`, the exponent may exceed `int` as long as the scale fits `int`.
    let scale = i64::try_from(fraction.len()).ok()?.checked_sub(exponent)?;
    let scale = i32::try_from(scale).ok()?;
    Some((format!("{integer}{fraction}"), scale))
}

/// Java's `DateTimeFormatter.ISO_LOCAL_DATE`: a year, `-MM-dd`, and a valid proleptic Gregorian
/// date. The year has four digits, five to ten with a `+` sign, or four to ten with a `-` sign, and
/// lies within Java's range of +-999,999,999. Negative zero is not a year.
fn is_iso_local_date(value: &str) -> bool {
    let (negative, year_widths, unsigned) = if let Some(unsigned) = value.strip_prefix('+') {
        (false, 5..=10, unsigned)
    } else if let Some(unsigned) = value.strip_prefix('-') {
        (true, 4..=10, unsigned)
    } else {
        (false, 4..=4, value)
    };
    let mut parts = unsigned.split('-');
    let (Some(year), Some(month), Some(day), None) =
        (parts.next(), parts.next(), parts.next(), parts.next())
    else {
        return false;
    };
    let (Some(year), Some(month), Some(day)) = (
        ascii_number(year, year_widths),
        ascii_number(month, 2..=2),
        ascii_number(day, 2..=2),
    ) else {
        return false;
    };
    if year > 999_999_999 || (negative && year == 0) {
        return false;
    }
    let leap_year = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => return false,
    };
    (1..=days_in_month).contains(&day)
}

/// Java's `DateTimeFormatter.ISO_LOCAL_DATE_TIME`: an ISO local date, `T` (in any case), and
/// `HH:mm`, optionally followed by `:ss` and a `.` with up to nine fractional digits. Like Java,
/// the second is at most 59, so a leap second is rejected.
fn is_iso_local_date_time(value: &str) -> bool {
    let Some((date, time)) = value.split_once(['T', 't']) else {
        return false;
    };
    let (time, fraction) = match time.split_once('.') {
        Some((time, fraction)) => (time, Some(fraction)),
        None => (time, None),
    };
    let fraction_ok =
        |fraction: Option<&str>| fraction.is_none_or(|f| ascii_number(f, 0..=9).is_some());
    let fields = time
        .split(':')
        .map(|field| ascii_number(field, 2..=2))
        .collect::<Option<Vec<_>>>();
    let time_ok = match fields.as_deref() {
        Some(&[hour, minute]) => fraction.is_none() && hour < 24 && minute < 60,
        Some(&[hour, minute, second]) => {
            fraction_ok(fraction) && hour < 24 && minute < 60 && second < 60
        }
        _ => false,
    };
    time_ok && is_iso_local_date(date)
}

/// Parses `value` as ASCII digits whose count is within `widths`. An empty `value` is `0` when
/// `widths` allows it.
fn ascii_number(value: &str, widths: RangeInclusive<usize>) -> Option<u64> {
    if !widths.contains(&value.len()) || !value.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    if value.is_empty() {
        return Some(0);
    }
    value.parse().ok()
}

/// Strips the offset of a Java `ISO_OFFSET_DATE_TIME` value and returns the local date-time. The
/// offset is `Z` (in any case) or `+HH:MM` / `-HH:MM`, optionally with `:ss`, of at most 18 hours.
/// Java's optional `[zone]` suffix is rejected; see [`validate_literal`].
fn split_offset(value: &str) -> Option<&str> {
    if let Some(local) = value.strip_suffix(['Z', 'z']) {
        return Some(local);
    }
    let index = value.rfind(['+', '-'])?;
    let (local, offset) = value.split_at(index);
    let parts = offset[1..]
        .split(':')
        .map(|part| {
            (part.len() == 2 && part.bytes().all(|b| b.is_ascii_digit()))
                .then(|| part.parse::<u32>().ok())
                .flatten()
        })
        .collect::<Option<Vec<_>>>()?;
    let seconds = match parts[..] {
        [hours, minutes] if minutes < 60 => hours * 3600 + minutes * 60,
        [hours, minutes, seconds] if minutes < 60 && seconds < 60 => {
            hours * 3600 + minutes * 60 + seconds
        }
        _ => return None,
    };
    // The date's own hyphens precede the `T`, so a valid offset follows it.
    (seconds <= 18 * 3600 && local.contains(['T', 't'])).then_some(local)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::schema::{ArrayType, StructField};

    fn decimal(precision: u8, scale: u8) -> DataType {
        DataType::decimal(precision, scale).unwrap()
    }

    #[rstest]
    #[case::single_quoted_string(DataType::STRING, "'abc'")]
    #[case::double_quoted_string(DataType::STRING, "\"CURRENT_TIMESTAMP()\"")]
    #[case::empty_string(DataType::STRING, "''")]
    #[case::quoted_binary(DataType::BINARY, "'abc'")]
    #[case::long(DataType::LONG, "-9223372036854775808")]
    #[case::quoted_integer(DataType::INTEGER, "'127'")]
    #[case::signed_integer(DataType::INTEGER, "+350")]
    #[case::short(DataType::SHORT, "32767")]
    #[case::float(DataType::FLOAT, "4.95")]
    #[case::float_suffix(DataType::FLOAT, "4.95f")]
    #[case::quoted_double(DataType::DOUBLE, "'3.21'")]
    #[case::double_exponent(DataType::DOUBLE, "-1.5e10")]
    #[case::double_nan(DataType::DOUBLE, "NaN")]
    #[case::double_infinity(DataType::DOUBLE, "-Infinity")]
    #[case::decimal(decimal(10, 2), "12345678.90")]
    #[case::decimal_fewer_digits(decimal(10, 5), "4.9")]
    #[case::decimal_exponent(decimal(3, 0), "1.2E+2")]
    #[case::decimal_leading_zeros(decimal(2, 1), "-007.5")]
    #[case::decimal_min_scale(decimal(1, 0), "1e2147483648")]
    #[case::boolean(DataType::BOOLEAN, "true")]
    #[case::boolean_any_value(DataType::BOOLEAN, "yes")]
    #[case::date(DataType::DATE, "2025-01-01")]
    #[case::quoted_date(DataType::DATE, "'2024-02-29'")]
    #[case::timestamp_utc(DataType::TIMESTAMP, "2025-01-01T00:00:00Z")]
    #[case::timestamp_offset(DataType::TIMESTAMP, "2025-01-01T10:15:30.123456+02:00")]
    #[case::timestamp_minutes(DataType::TIMESTAMP, "2025-01-01T10:15-05:30")]
    #[case::timestamp_lowercase(DataType::TIMESTAMP, "2025-01-01t00:00:00z")]
    #[case::timestamp_offset_seconds(DataType::TIMESTAMP, "2025-01-01T00:00:00-18:00:00")]
    #[case::timestamp_ntz(DataType::TIMESTAMP_NTZ, "2025-01-01T00:00:00")]
    #[case::timestamp_ntz_nanos(DataType::TIMESTAMP_NTZ, "'2025-01-01T23:59:59.123456789'")]
    #[case::timestamp_ntz_empty_fraction(DataType::TIMESTAMP_NTZ, "2025-01-01T00:00:00.")]
    #[case::timestamp_empty_fraction(DataType::TIMESTAMP, "2025-01-01T00:00:00.Z")]
    #[case::timestamp_extended_year(DataType::TIMESTAMP, "+10000-01-01T00:00:00Z")]
    #[case::timestamp_ntz_negative_year(DataType::TIMESTAMP_NTZ, "-0001-01-01T00:00")]
    #[case::date_extended_year(DataType::DATE, "+10000-01-01")]
    #[case::date_padded_year(DataType::DATE, "+00001-01-01")]
    #[case::date_max_year(DataType::DATE, "+999999999-12-31")]
    #[case::date_min_year(DataType::DATE, "-999999999-01-01")]
    #[case::date_year_zero(DataType::DATE, "0000-01-01")]
    #[case::date_extended_leap_day(DataType::DATE, "+12000-02-29")]
    #[case::double_hex(DataType::DOUBLE, "0x1p3")]
    #[case::double_hex_fraction(DataType::DOUBLE, "-0X1.8P-1d")]
    #[case::float_hex_no_integer(DataType::FLOAT, "0x.8p1f")]
    #[case::double_out_of_range(DataType::DOUBLE, "1e400")]
    fn validate_literal_accepts(#[case] data_type: DataType, #[case] value: &str) {
        assert_eq!(validate_literal(&data_type, value), Ok(()));
    }

    #[rstest]
    #[case::unquoted_string(DataType::STRING, "CURRENT_TIMESTAMP()")]
    #[case::single_quote(DataType::STRING, "'")]
    #[case::mismatched_quotes(DataType::STRING, "'abc\"")]
    #[case::unquoted_binary(DataType::BINARY, "X'01'")]
    #[case::null(DataType::INTEGER, "NULL")]
    #[case::integer_fraction(DataType::INTEGER, "313.55")]
    #[case::integer_overflow(DataType::INTEGER, "2147483648")]
    #[case::short_overflow(DataType::SHORT, "32768")]
    #[case::long_expression(DataType::LONG, "1 + 1")]
    #[case::double_text(DataType::DOUBLE, "Good boy")]
    #[case::double_rust_only_spelling(DataType::DOUBLE, "inf")]
    #[case::decimal_precision(decimal(10, 5), "234243243243243234.234")]
    #[case::decimal_scale(decimal(10, 2), "1.234")]
    #[case::decimal_text(decimal(10, 2), "abc")]
    #[case::decimal_empty_exponent(decimal(10, 2), "1e")]
    #[case::decimal_scale_below_int(decimal(10, 2), "1e2147483649")]
    #[case::date_slashes(DataType::DATE, "2022/01/05")]
    #[case::date_invalid_day(DataType::DATE, "2023-02-29")]
    #[case::date_short_month(DataType::DATE, "2023-1-05")]
    #[case::date_unsigned_extended_year(DataType::DATE, "10000-01-01")]
    #[case::date_signed_four_digit_year(DataType::DATE, "+2025-01-01")]
    #[case::date_negative_zero_year(DataType::DATE, "-0000-01-01")]
    #[case::date_year_too_large(DataType::DATE, "+1000000000-01-01")]
    #[case::date_extended_non_leap_day(DataType::DATE, "+12100-02-29")]
    #[case::date_trailing_part(DataType::DATE, "2025-01-01-01")]
    #[case::timestamp_zone_suffix(DataType::TIMESTAMP, "2025-01-01T00:00:00Z[UTC]")]
    #[case::integer_non_ascii_digit(DataType::INTEGER, "\u{661}")]
    #[case::decimal_non_ascii_digit(decimal(10, 2), "\u{661}")]
    #[case::double_hex_without_exponent(DataType::DOUBLE, "0x1")]
    #[case::double_hex_without_digits(DataType::DOUBLE, "0xp1")]
    #[case::double_hex_empty_exponent(DataType::DOUBLE, "0x1p")]
    #[case::double_two_signs(DataType::DOUBLE, "+-1")]
    #[case::double_hex_two_signs(DataType::DOUBLE, "+-0x1p3")]
    #[case::timestamp_date_only(DataType::TIMESTAMP, "2025-01-01")]
    #[case::timestamp_without_offset(DataType::TIMESTAMP, "2025-01-01T00:00:00")]
    #[case::timestamp_space(DataType::TIMESTAMP, "2025-01-01 00:00:00Z")]
    #[case::timestamp_bad_offset(DataType::TIMESTAMP, "2025-01-01T00:00:00+2:00")]
    #[case::timestamp_offset_too_large(DataType::TIMESTAMP, "2025-01-01T00:00:00+18:30")]
    #[case::timestamp_ntz_offset(DataType::TIMESTAMP_NTZ, "2025-01-01T00:00:00+02:00")]
    #[case::timestamp_ntz_minutes_fraction(DataType::TIMESTAMP_NTZ, "2025-01-01T00:00.")]
    #[case::timestamp_ntz_long_fraction(DataType::TIMESTAMP_NTZ, "2025-01-01T00:00:00.1234567890")]
    #[case::timestamp_ntz_bad_hour(DataType::TIMESTAMP_NTZ, "2025-01-01T24:00:00")]
    #[case::timestamp_ntz_leap_second(DataType::TIMESTAMP_NTZ, "2025-01-01T23:59:60")]
    #[case::timestamp_ntz_leap_second_fraction(DataType::TIMESTAMP_NTZ, "2025-01-01T23:59:60.5")]
    #[case::timestamp_leap_second(DataType::TIMESTAMP, "2025-06-30T23:59:60Z")]
    fn validate_literal_rejects_non_literals(#[case] data_type: DataType, #[case] value: &str) {
        assert_eq!(
            validate_literal(&data_type, value),
            Err(LiteralError::NotLiteral)
        );
    }

    #[rstest]
    #[case::byte(DataType::BYTE)]
    #[case::array(ArrayType::new(DataType::INTEGER, true).into())]
    #[case::struct_type(StructType::new_unchecked([StructField::nullable("a", DataType::INTEGER)]).into())]
    fn validate_literal_rejects_unsupported_types(#[case] data_type: DataType) {
        assert_eq!(
            validate_literal(&data_type, "1"),
            Err(LiteralError::UnsupportedType)
        );
    }
}
