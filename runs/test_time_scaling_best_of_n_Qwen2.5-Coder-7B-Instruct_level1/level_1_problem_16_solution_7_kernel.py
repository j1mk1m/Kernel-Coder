if __name__ == "__main__":
    model = ModelNew().cuda()
    A, B = get_inputs()
    C = model.forward(A.cuda(), B.cuda())
    print(C.shape)