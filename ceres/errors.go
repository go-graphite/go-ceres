package ceres

import (
	"errors"
)

var ErrInvalidFrom = errors.New("time interval is not valid: from is after until")
