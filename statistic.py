class Statistic():
    @staticmethod
    def mean(args):
        return sum(args)/len(args)

    @staticmethod
    def median(args):
        n = int(len(args)/2)
        list = sorted(args)
        return list[n] if len(args) % 2 != 0 else list[n] + (list[n] - 1)

    @staticmethod
    def quantile(args, percentage):
        quantiles = {
            0.25    : sorted(args)[int(len(args)/4)],
            0.5     : sorted(args)[int(len(args)/2)],
            0.75    : sorted(args)[int(3 * len(args)/4)]
        }
        return quantiles[percentage]

    @staticmethod
    def var(args):
        stock = []
        for arg in args:
            stock.append((arg - Statistic.mean(args)) ** 2)
        return sum(stock)/len(args)

    @staticmethod
    def std(args):
        return Statistic.var(args)**(1/2)