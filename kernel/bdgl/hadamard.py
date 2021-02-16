from numpy import array, zeros
from numpy.linalg import inv

def _mm256_hadd_epi16(a, b):
	dst = zeros(16, dtype=long)
	dst[0::8] = a[0::8] + a[1::8] 
	dst[1::8] = a[2::8] + a[3::8] 
	dst[2::8] = a[4::8] + a[5::8] 
	dst[3::8] = a[6::8] + a[7::8] 
	dst[4::8] = b[0::8] + b[1::8] 
	dst[5::8] = b[2::8] + b[3::8] 
	dst[6::8] = b[4::8] + b[5::8] 
	dst[7::8] = b[6::8] + b[7::8] 
	return dst


s1 = array(8*[1, -1], dtype=long)
s4 = array(8*[1]+8*[-1], dtype=long)

def hadamard16(x):
	# a = _mm256_hadd_epi16(a, a)
	# return _mm256_hadd_epi16(a, a)
	a = 1*x
	a[:8],a[8:] = x[8:],x[:8]
	x = x*s4
	a += x	

	b = a*s1
	a = _mm256_hadd_epi16(a, b)	
	b = a*s1
	a = _mm256_hadd_epi16(a, b)	
	b = a*s1
	a = _mm256_hadd_epi16(a, b)	
	return a


def hadamard32(x):
	r = zeros(32, dtype=int)
	a = hadamard16(x[:16])
	b = hadamard16(x[16:])
	r[:16]= a + b
	r[16:]= a - b

	return r


def hadamard32_mat():
	M = zeros((32, 32),dtype=int)
	for i in range(32):
		v = zeros(32, dtype=int)
		v[i] = 1
		M[i] = hadamard32(v)
	return M

