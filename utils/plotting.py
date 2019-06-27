import matplotlib.pyplot as plt

def plot_profile(profiles):
    plt.style.use('bmh')
    plt.figure()
    for e in profiles:
        plt.plot(profiles[e], label = e)
    plt.legend()
    plt.show()


class LBFAPlot:
    @staticmethod
    def plot1():
        plt.style.use('bmh')
        pass

    @staticmethod
    def plot2():
        plt.style.use('bmh')
        pass


class DQNPlot:
    @staticmethod
    def plot1():
        plt.style.use('bmh')
        pass

    @staticmethod
    def plot2():
        plt.style.use('bmh')
        pass


class DPGPlot:
    @staticmethod
    def plot1():
        plt.style.use('bmh')
        pass

    @staticmethod
    def plot2():
        plt.style.use('bmh')
        pass