package ceres

import (
	"math"
)

//	Implementation of modulo that works like Python
//	Thanks @timmow for this
func mod(a, b int) int {
	return a - (b * int(math.Floor(float64(a)/float64(b))))
}
