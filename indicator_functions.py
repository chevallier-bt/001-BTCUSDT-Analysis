import copy


def get_sma(items, *args):
    """
    get_sma() function will return the simple moving average for the list "items" with period "n".
    Use of args here is redundant, but I was planning on adding more to this function later.
    """
    items = copy.copy(items)
    n = args[0]
    return items.rolling(n).mean()


def get_std(items, *args):
    """
    get_sd() function will return the standard deviation for the list "items" with period "n".
    Use of args here is redundant, but I was planning on adding more to this function later.
    """
    items = copy.copy(items)
    n = args[0]
    return items.rolling(n).std()
