import cProfile, pstats, io


def profile(func):
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream = s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval
    return inner