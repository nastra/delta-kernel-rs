//! Validates the column defaults of a replacement schema, mirroring Delta Kernel Java's
//! `ColumnDefaults.validateSchema` and `ColumnDefaults.validateLiteral`.

use chrono::{NaiveDate, NaiveTime};

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

/// Java's `Double.parseDouble` for decimal notation: surrounding whitespace, an optional `f`,
/// `F`, `d`, or `D` suffix, and the special values `NaN` and `Infinity`. Hexadecimal notation is
/// not accepted.
fn is_java_floating_point(value: &str) -> bool {
    let value = value.trim_matches(|c: char| c <= ' ');
    let unsigned = value.strip_prefix(['+', '-']).unwrap_or(value);
    if unsigned == "Infinity" || unsigned == "NaN" {
        return true;
    }
    let number = value.strip_suffix(['f', 'F', 'd', 'D']).unwrap_or(value);
    number
        .chars()
        .all(|c| c.is_ascii_digit() || matches!(c, '+' | '-' | '.' | 'e' | 'E'))
        && number.parse::<f64>().is_ok()
}

/// Java's `new BigDecimal(value)` followed by the check that the scale and precision fit
/// `decimal`. A negative scale (from a positive exponent) always fits.
fn fits_decimal(value: &str, decimal: &DecimalType) -> bool {
    let Some((digits, scale)) = parse_big_decimal(value) else {
        return false;
    };
    let significant = digits.trim_start_matches('0').len().max(1);
    scale <= i64::from(decimal.scale()) && significant <= usize::from(decimal.precision())
}

/// Parses `[+-]digits[.digits][(e|E)[+-]digits]` into the unscaled digits and the scale, as Java's
/// `BigDecimal(String)` constructor does.
fn parse_big_decimal(value: &str) -> Option<(String, i64)> {
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
    let scale = i64::try_from(fraction.len()).ok()?.checked_sub(exponent)?;
    Some((format!("{integer}{fraction}"), scale))
}

/// Java's `DateTimeFormatter.ISO_LOCAL_DATE`: `yyyy-MM-dd` with a valid calendar date.
fn is_iso_local_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes
            .iter()
            .enumerate()
            .all(|(i, b)| i == 4 || i == 7 || b.is_ascii_digit())
        && NaiveDate::parse_from_str(value, "%Y-%m-%d").is_ok()
}

/// Java's `DateTimeFormatter.ISO_LOCAL_DATE_TIME`: an ISO local date, `T` (in any case), and
/// `HH:mm`, optionally followed by `:ss` and one to nine fractional digits.
fn is_iso_local_date_time(value: &str) -> bool {
    let Some((date, time)) = value.split_once(['T', 't']) else {
        return false;
    };
    let (time, fraction) = match time.split_once('.') {
        Some((time, fraction)) => (time, Some(fraction)),
        None => (time, None),
    };
    let format = match time.len() {
        5 => "%H:%M",
        8 => "%H:%M:%S",
        _ => return false,
    };
    let fraction_ok = fraction.is_none_or(|fraction| {
        time.len() == 8
            && (1..=9).contains(&fraction.len())
            && fraction.bytes().all(|b| b.is_ascii_digit())
    });
    is_iso_local_date(date)
        && fraction_ok
        && time.bytes().enumerate().all(|(i, b)| {
            if i % 3 == 2 {
                b == b':'
            } else {
                b.is_ascii_digit()
            }
        })
        && NaiveTime::parse_from_str(time, format).is_ok()
}

/// Strips the offset of a Java `ISO_OFFSET_DATE_TIME` value and returns the local date-time. The
/// offset is `Z` (in any case) or `+HH:MM` / `-HH:MM`, optionally with `:ss`, of at most 18 hours.
/// Java's optional `[zone]` suffix is not accepted.
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
    #[case::date_slashes(DataType::DATE, "2022/01/05")]
    #[case::date_invalid_day(DataType::DATE, "2023-02-29")]
    #[case::date_short_month(DataType::DATE, "2023-1-05")]
    #[case::timestamp_date_only(DataType::TIMESTAMP, "2025-01-01")]
    #[case::timestamp_without_offset(DataType::TIMESTAMP, "2025-01-01T00:00:00")]
    #[case::timestamp_space(DataType::TIMESTAMP, "2025-01-01 00:00:00Z")]
    #[case::timestamp_bad_offset(DataType::TIMESTAMP, "2025-01-01T00:00:00+2:00")]
    #[case::timestamp_offset_too_large(DataType::TIMESTAMP, "2025-01-01T00:00:00+18:30")]
    #[case::timestamp_ntz_offset(DataType::TIMESTAMP_NTZ, "2025-01-01T00:00:00+02:00")]
    #[case::timestamp_ntz_empty_fraction(DataType::TIMESTAMP_NTZ, "2025-01-01T00:00:00.")]
    #[case::timestamp_ntz_long_fraction(DataType::TIMESTAMP_NTZ, "2025-01-01T00:00:00.1234567890")]
    #[case::timestamp_ntz_bad_hour(DataType::TIMESTAMP_NTZ, "2025-01-01T24:00:00")]
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
