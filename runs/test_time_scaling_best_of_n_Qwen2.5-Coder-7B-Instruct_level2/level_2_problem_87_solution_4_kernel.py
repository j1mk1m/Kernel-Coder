class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.my_custom_function = custom_code.my_custom_function

    def forward(self, x):
        self.my_custom_function(x)
        return x