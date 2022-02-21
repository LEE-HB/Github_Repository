def generator():
    i = 0
    while 1:
        i += 1
        yield i


for x in generator():
    print(x, type(x))

    if x > 5:
        break