def clr(iteration, max_lr, base_lr, stepsize):
    cycle = stepsize * 2
    iteration %= cycle
    width = max_lr - base_lr
    lr = abs(2 * width / cycle * iteration - width) + base_lr
    return lr
