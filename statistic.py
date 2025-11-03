class Statistic():
    @staticmethod
    def mean(args):
        return sum(args)/len(args)

    @staticmethod
    def median(args):
        sorted_args = sorted(args)
        n = len(sorted_args)
        if n % 2 == 0:
            return (sorted_args[n//2 - 1] + sorted_args[n//2]) / 2
        else:
            return sorted_args[n//2]

    @staticmethod
    def quantile(args, percentage):
        sorted_args = sorted(args)
        n = len(sorted_args)
        
        if percentage == 0.25:
            index = int(n * 0.25)
            if index >= n:
                index = n - 1
            return sorted_args[index]
        elif percentage == 0.5:
            # Use the median calculation for 50% quantile
            return Statistic.median(sorted_args)
        elif percentage == 0.75:
            index = int(n * 0.75)
            if index >= n:
                index = n - 1
            return sorted_args[index]
        else:
            raise ValueError("Only 0.25, 0.5, and 0.75 quantiles are supported")

    @staticmethod
    def variance(args):
        stock = []
        for arg in args:
            stock.append((arg - Statistic.mean(args)) ** 2)
        return sum(stock)/len(args)

    @staticmethod
    def std(args):
        return Statistic.variance(args)**(1/2)

    @staticmethod
    def min(args):
        """Find minimum value without using built-in min()"""
        if len(args) == 0:
            raise ValueError("Cannot find min of empty sequence")
        
        min_val = args[0]
        for val in args[1:]:
            if val < min_val:
                min_val = val
        return min_val

    @staticmethod
    def max(args):
        """Find maximum value without using built-in max()"""
        if len(args) == 0:
            raise ValueError("Cannot find max of empty sequence")
        
        max_val = args[0]
        for val in args[1:]:
            if val > max_val:
                max_val = val
        return max_val