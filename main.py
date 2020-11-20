import lib


if __name__ == '__main__':
    alphabet = ['add', 'mul', 'sub', 'div', 'sin', 'cos', 'x', 'const']
    decoder = lib.LSTMDecoder(alphabet)

    out = decoder.forward()
    prefix = [alphabet[x] for x in out]

    print(prefix)
    print(lib.prefix_to_infix(prefix))
