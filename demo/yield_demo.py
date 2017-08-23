def fab(max):
    n, a, b = 0, 0, 1
    while n < max:
        yield b
        # print b 
        a, b = b, a + b
        n = n + 1


g = fab(5)
for n in g:
    print(n)
