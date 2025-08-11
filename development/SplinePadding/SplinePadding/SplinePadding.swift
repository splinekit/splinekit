import Foundation

@_cdecl("get_samples_to_coeff_p")
public func get_samples_to_coeff_p (
	_ data: UnsafeMutablePointer<Double>?,
	_ dataCount: Int32,
	_ pole: UnsafeMutablePointer<Double>?,
	_ poleCount: Int32
) {
	guard nil != data && 0 < dataCount
	else {
		return
	}
	guard nil != pole && 0 <= poleCount
	else {
		return
	}
	let P = Int(dataCount)
	if 0 != Int(poleCount) && 1 != P {
		for q in 0 ..< Int(poleCount) {
			let z = pole![q]
			var sigma = data![0]
			var zeta = z
			var k = 1
			while k < P && 0.0 != zeta {
				sigma += zeta * data![P - k]
				zeta *= z
				k += 1
			}
			data![0] = sigma / (1.0 - zeta)
			for k in 1 ..< P {
				data![k] += z * data![k - 1]
			}
			sigma = data![P - 1]
			zeta = z
			k = 0
			while k < P - 1 && 0.0 != zeta {
				sigma += zeta * data![k]
				zeta *= z
				k += 1
			}
			let z12 = (1.0 - z) * (1.0 - z)
			data![P - 1] = z12 * sigma / (1.0 - zeta)
			for k in 1 ..< P {
				data![P - 1 - k] = z * data![P - k] + z12 * data![P - 1 - k]
			}
		}
	}
}

@_cdecl("get_samples_to_coeff_n")
public func get_samples_to_coeff_n (
	_ data: UnsafeMutablePointer<Double>?,
	_ dataCount: Int32,
	_ pole: UnsafeMutablePointer<Double>?,
	_ poleCount: Int32
) {
	guard nil != data && 0 < dataCount
	else {
		return
	}
	guard nil != pole && 0 <= poleCount
	else {
		return
	}
	let K = Int(dataCount)
	if 0 != Int(poleCount) && 1 != K {
		for q in 0 ..< Int(poleCount) {
			let z = pole![q]
			var sigma1 = data![0]
			var sigma2 = data![K - 1]
			var zeta = z
			var k = 1
			while k < K - 1 && 0.0 != zeta {
				sigma1 += zeta * data![k]
				sigma2 += zeta * data![K - 1 - k]
				zeta *= z
				k += 1
			}
			data![0] = (sigma1 + zeta * sigma2) / (1.0 - zeta * zeta)
			for k in 1 ..< K {
				data![k] += z * data![k - 1]
			}
			let z12 = (1.0 - z) * (1.0 - z)
			data![K - 1] = z12 * (z * data![K - 2] + data![K - 1]) / (1.0 - z * z)
			for k in 1 ..< K {
				data![K - 1 - k] = z * data![K - k] + z12 * data![K - 1 - k]
			}
		}
	}
}

@_cdecl("get_samples_to_coeff_w")
public func get_samples_to_coeff_w (
	_ data: UnsafeMutablePointer<Double>?,
	_ dataCount: Int32,
	_ pole: UnsafeMutablePointer<Double>?,
	_ poleCount: Int32
) {
	guard nil != data && 0 < dataCount
	else {
		return
	}
	guard nil != pole && 0 <= poleCount
	else {
		return
	}
	let K = Int(dataCount)
	if 0 != Int(poleCount) && 1 != K {
		for q in 0 ..< Int(poleCount) {
			let z = pole![q]
			var sigma1 = 0.0
			var sigma2 = 0.0
			var zeta = 1.0
			var k = 0
			while k < K && 0.0 != zeta {
				sigma1 += zeta * data![k]
				sigma2 += zeta * data![K - 1 - k]
				zeta *= z
				k += 1
			}
			data![0] += z * (sigma1 + zeta * sigma2) / (1.0 - zeta * zeta)
			for k in 1 ..< K {
				data![k] += z * data![k - 1]
			}
			let z12 = (1.0 - z) * (1.0 - z)
			data![K - 1] *= 1.0 - z
			for k in 1 ..< K {
				data![K - 1 - k] = z * data![K - k] + z12 * data![K - 1 - k]
			}
		}
	}
}

@_cdecl("get_samples_to_coeff_a")
public func get_samples_to_coeff_a (
	_ data: UnsafeMutablePointer<Double>?,
	_ dataCount: Int32,
	_ pole: UnsafeMutablePointer<Double>?,
	_ poleCount: Int32
) {
	guard nil != data && 0 < dataCount
	else {
		return
	}
	guard nil != pole && 0 <= poleCount
	else {
		return
	}
	let K = Int(dataCount)
	if 0 != Int(poleCount) && 1 != K {
		for q in 0 ..< Int(poleCount) {
			let z = pole![q]
			var sigma1 = 0.0
			var sigma2 = 0.0
			var zeta = z
			var k = 1
			while k < K - 1 && 0.0 != zeta {
				sigma1 += zeta * data![k]
				sigma2 += zeta * data![K - 1 - k]
				zeta *= z
				k += 1
			}
			data![0] = ((data![0] - zeta * data![K - 1]) * (1.0 + z) / (1.0 - z) - sigma1 + zeta * sigma2) / (1.0 - zeta * zeta)
			for k in 1 ..< K {
				data![k] += z * data![k - 1]
			}
			let z12 = (1.0 - z) * (1.0 - z)
			data![K - 1] -= z * data![K - 2]
			for k in 1 ..< K {
				data![K - 1 - k] = z * data![K - k] + z12 * data![K - 1 - k]
			}
		}
	}
}

@_cdecl("get_samples_to_coeff_np")
public func get_samples_to_coeff_np (
	_ data: UnsafeMutablePointer<Double>?,
	_ dataCount: Int32,
	_ pole: UnsafeMutablePointer<Double>?,
	_ poleCount: Int32
) {
	guard nil != data && 0 < dataCount
	else {
		return
	}
	guard nil != pole && 0 <= poleCount
	else {
		return
	}
	let K = Int(dataCount)
	for q in 0 ..< Int(poleCount) {
		let z = pole![q]
		var sigma1 = data![0]
		var sigma2 = data![K - 1]
		var zeta = z
		var k = 1
		while k < K && 0.0 != zeta {
			sigma1 += zeta * data![k]
			sigma2 += zeta * data![K - 1 - k]
			zeta *= z
			k += 1
		}
		let Z0 = z / (1.0 + zeta)
		sigma1 *= Z0
		sigma2 *= Z0
		data![0] -= sigma2
		for k in 1 ..< K {
			data![k] += z * data![k - 1]
		}
		zeta *= zeta
		let z12 = (1.0 - z) * (1.0 - z)
		let Z1 = (1.0 - z) / (1.0 + z)
		data![K - 1] *= (1.0 + zeta) * Z1
		data![K - 1] -= (sigma2 * zeta / z + sigma1) * Z1
		for k in 1 ..< K {
			data![K - 1 - k] = z * data![K - k] + z12 * data![K - 1 - k]
		}
	}
}

@_cdecl("get_samples_to_coeff_nn")
public func get_samples_to_coeff_nn (
	_ data: UnsafeMutablePointer<Double>?,
	_ dataCount: Int32,
	_ pole: UnsafeMutablePointer<Double>?,
	_ poleCount: Int32
) {
	guard nil != data && 0 < dataCount
	else {
		return
	}
	guard nil != pole && 0 <= poleCount
	else {
		return
	}
	let K = Int(dataCount)
	for q in 0 ..< Int(poleCount) {
		let z = pole![q]
		var sigma1 = 0.0
		var sigma2 = 0.0
		var zeta = 1.0
		var k = 0
		while k < K && 0.0 != zeta {
			sigma1 += zeta * data![k]
			sigma2 += zeta * data![K - 1 - k]
			zeta *= z
			k += 1
		}
		zeta *= z
		data![0] -= (sigma1 - zeta * sigma2) * z * z / (1.0 - zeta * zeta)
		for k in 1 ..< K {
			data![k] += z * data![k - 1]
		}
		let z12 = (1.0 - z) * (1.0 - z)
		data![K - 1] *= z12
		for k in 1 ..< K {
			data![K - 1 - k] = z * data![K - k] + z12 * data![K - 1 - k]
		}
	}
}

@_cdecl("get_samples_to_coeff_nw")
public func get_samples_to_coeff_nw (
	_ data: UnsafeMutablePointer<Double>?,
	_ dataCount: Int32,
	_ pole: UnsafeMutablePointer<Double>?,
	_ poleCount: Int32
) {
	guard nil != data && 0 < dataCount
	else {
		return
	}
	guard nil != pole && 0 <= poleCount
	else {
		return
	}
	let K = Int(dataCount)
	for q in 0 ..< Int(poleCount) {
		let z = pole![q]
		var sigma1 = 0.0
		var sigma2 = 0.0
		var zeta = 1.0
		var k = 0
		while k < K && 0.0 != zeta {
			sigma1 += zeta * data![k]
			sigma2 += zeta * data![K - 1 - k]
			zeta *= z
			k += 1
		}
		data![0] -= (sigma1 - zeta * sigma2) * z / (1.0 - zeta * zeta)
		for k in 1 ..< K {
			data![k] += z * data![k - 1]
		}
		let z12 = (1.0 - z) * (1.0 - z)
		data![K - 1] *= z12 / (1.0 + z)
		for k in 1 ..< K {
			data![K - 1 - k] = z * data![K - k] + z12 * data![K - 1 - k]
		}
	}
}
