import numpy as np
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def func(x, y):
    ans = 0
    for i in range(n):
        xx = point_list[i].x
        yy = point_list[i].y
        p = xx - x
        q = yy - y
        ans += np.sqrt(p * p + q * q)
    return ans

if __name__ == '__main__':
    n = 105
    T = 3000
    dT = 0.995
    eps = 1e-8
    np.random.seed(220)
    point_list = [Point(np.random.randint(-100, 100), np.random.randint(-100, 100)) for _ in range(n)]
    p0 = Point(0, 0)
    f = func(p0.x, p0.y)
    while(T > eps):
        dx = (p0.x + np.random.randint(-100, 100) * T) % 100
        dy = (p0.y + np.random.randint(-100, 100) * T) % 100
        df = func(dx, dy)
        if df < f:
            p0.x = dx
            p0.y = dy
            f = df
        elif np.exp((f - df) / T) > np.random.uniform(0, 1):
            p0.x = dx
            p0.y = dy
            f = df
        T *= dT
    print(p0.x, p0.y)
