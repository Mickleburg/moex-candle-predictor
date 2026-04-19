package models

import (
	"fmt"
	"strings"
	"time"
)

var flexibleTimeLayouts = []string{
	time.RFC3339Nano,
	time.RFC3339,
	"2006-01-02T15:04:05.999999999",
	"2006-01-02T15:04:05.999999",
	"2006-01-02T15:04:05",
	"2006-01-02 15:04:05",
	"2006-01-02",
}

func ParseFlexibleTime(value string) (time.Time, error) {
	trimmed := strings.TrimSpace(value)
	if trimmed == "" {
		return time.Time{}, fmt.Errorf("empty timestamp")
	}

	for _, layout := range flexibleTimeLayouts {
		if parsed, err := time.Parse(layout, trimmed); err == nil {
			switch layout {
			case time.RFC3339Nano, time.RFC3339:
				return parsed.UTC(), nil
			default:
				// Layouts without an explicit zone are interpreted as UTC.
				return time.Date(
					parsed.Year(), parsed.Month(), parsed.Day(),
					parsed.Hour(), parsed.Minute(), parsed.Second(), parsed.Nanosecond(),
					time.UTC,
				), nil
			}
		}
	}

	return time.Time{}, fmt.Errorf("unsupported timestamp format %q", value)
}
