import src.settings as settings

def printer(*args,verb=1):
    if settings.verb >= verb:
        print(args)