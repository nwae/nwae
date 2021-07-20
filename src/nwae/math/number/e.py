# -*- coding: utf-8 -*-


class e:

    def __init__(self):
        return

    def method_limit(self, x=99999999):
        return (1 + 1/x) ** x

    def method_limit_diff(self, x):
        return self.method_limit(x=x) - self.method_limit(x=x-1)


if __name__ == '__main__':
    print(e().method_limit())
    start = 1000
    step = 100
    for i in range(start, 100*start, step):
        print((i+1), '. val=', e().method_limit(x=i), ', diff=', e().method_limit_diff(x=i))
    exit(0)
