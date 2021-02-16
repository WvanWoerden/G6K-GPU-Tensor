from numpy import array, zeros
from hadamard import hadamard32_mat 
from numpy.random import randint
import sys

N = int(sys.argv[1])

mixmask0 = [
		0xFFFF, 0xFFFF, 0x0000, 0xFFFF, 0xFFFF, 0x0000, 0xFFFF, 0xFFFF, 
  		0xFFFF, 0xFFFF, 0xFFFF, 0x0000, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF]

mixmask0 = array(mixmask0)[::-1]%2

permutation1 =  [
				0x0F,   0x0E,   0x07,   0x06,   0x01,   0x00,   0x09,   0x08, 
				0x0B,   0x0A,   0x0D,   0x0C,   0x05,   0x04,   0x03,   0x02,
				0x07,   0x06,   0x0F,   0x0E,   0x05,   0x04,   0x03,   0x02,
				0x0B,   0x0A,   0x09,   0x08,   0x0D,   0x0C,   0x01,   0x00]

permutation1 = array(permutation1)[0::2][::-1]/2
permutation1[8:]+=8

sign_shuffle = [
  0xFFFF, 0xFFFF, 0xFFFF, 0x0001, 0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
  0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0x0001, 0xFFFF, 0xFFFF]

sign_shuffle =  2 - ((array(sign_shuffle[::-1])) % 4)

H32 = hadamard32_mat()

def m256_permute_epi16_2f(v, mixmask):
	v0 = zeros(16, dtype=long)
	v1 = zeros(16, dtype=long)
	v2 = zeros(16, dtype=long)
	tmp = zeros(16, dtype=long)
	v0[:] = 1*v[:16]
	v1[:] = 1*v[16:32]
	v2[:N-32] = v[32-N:]

	v1[0:4], v1[4:8], v1[8:12], v1[12:16] = (1*v1[12:16], 1*v1[0:4], 1*v1[4:8], 1*v1[8:12])
	v0 = array([1*v0[permutation1[i]] for i in range(16)])

	for i in range(16):
		if mixmask[i]:
			v0[i], v1[i] = 1*v1[i], 1*v0[i]

	v0 *= sign_shuffle

	for i in range(N-32):
		v1[i], v2[i] = 1*v2[i], 1*v1[i]

	v[:16] = v0[:]
	v[16:32] = v1[:]
	v[32-N:] = v2[:N-32]
	return v


def extract_vec(v):
	V = zeros((N, 32), dtype=long)
	for i in range(32):
		V[abs(v[i]) - 1] = H32[i] * (1 if v[i] > 0 else -1)
	return V.transpose()


def score(V, D):
	s = 0
	for W in D:
		A = V.dot(W.transpose())
		A = abs(A*A*A)
		s += sum(sum(A))
	return s


def generate_mixsequence(seqlen, trials, filename=None):
	if filename is not None:
		filee = open(filename,"w")
	else:
		filee = None

	v = array(range(N))+1
	v0 = 1*v

	D = []
	scores = []
	D.append(extract_vec(v))
	total_score = 0

	for it in range(seqlen):
		best = None
		for trial in range(trials):
			mixmask = randint(2, size=32)
			w = m256_permute_epi16_2f(v, mixmask)
			s = - score(extract_vec(w), D)
			# print s, 
			best = max(best, (s, 1*w, mixmask))

		(s, w, mixmask) = best
		total_score -= s

		scores.append(total_score)
		v = 1*w
		D.append(extract_vec(v))
		if filename is not None:
			for x in mixmask:
				print >>filee, x,
			print >>filee, ""
	
	return scores

generate_mixsequence(256, 16, "mix_sequence_%d.dat"%N)


# M = []
# for trials in [1,2,4,8,16,32,64,128,256]:
# 	print "%4d :"%trials,
# 	L = generate_mixsequence(100, trials)
# 	for x in L[::5]:
# 		print "%.4e "%x,
# 	print

# print array(M)
