

def get_plot_params():
    """Create the plot parameters used in the plotting 
    all the figures in the paper
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import rc
    # mpl.rcParams['text.usetex']=True
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = 'Arial'
    rc('text.latex', preamble=r'\usepackage{cmbright}')
    rc('text.latex', preamble=r'\usepackage{color}')
    plt.rcParams['font.size'] = 18
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16

    plt.rcParams['xtick.major.size'] = 10
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['xtick.minor.size'] = 5
    plt.rcParams['xtick.minor.width'] = 2
    plt.rcParams['ytick.major.size'] = 10
    plt.rcParams['ytick.major.width'] = 2
    plt.rcParams['ytick.minor.size'] = 5
    plt.rcParams['ytick.minor.width'] = 2

    plt.rcParams['axes.labelsize'] = 22


    COLOR = 'k'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    plt.rcParams['xtick.color'] = COLOR
    plt.rcParams['ytick.color'] = COLOR


