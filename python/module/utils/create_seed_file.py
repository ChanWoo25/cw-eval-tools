from pathlib import Path

t = 0.5
xs = [71.0, 138.0, 205.0, 272.0, 339.0, 406.0]
ys = [45.0, 112.0, 179.0, 246.0, 313.0]

fn = Path('./seeds.txt')
with open(fn, 'w') as f:
    for i in range(len(xs)):
        for j in range(len(ys)):
            index = i * len(ys) + j
            x = xs[i]
            y = ys[j]
            line = '%.2f,%.1f,%.1f,0.0,%d\n'%(t, x, y, index)
            f.write(line)
            # print(line)

