def count(N):
	ret = 0
	for i in range(N):
		for j in range(i+1,N):
			ret += 1
	return ret

for i in range(1,10):
	print(count(i))
