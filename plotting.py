# ana.rebeka.kamsek@ki.si, 2022

import numpy as np
import matplotlib.pyplot as plt


def individual_histograms(data, variable_list=None,
                          no_of_bins=20, set_width=0.9, alpha=0.6, color='darkblue', fontsize=16):
    """Plots number histograms for the chosen variables.

    Takes the data containing particle areas, perimeters, circularities, equivalent sphere diameters,
    and equivalent sphere volumes, and a list of variable indices to be used. Displays the histograms
    side by side. Different plot parameters can be passed.
    :param data: ndarray of shape (number of particles, 5)
    :param variable_list: a list of ints to create a subset from the data
    :param no_of_bins: number of histogram bins
    :param set_width: bin width
    :param alpha: transparency of the columns
    :param color: color of the columns
    :param fontsize: font size
    """

    # select the data for plotting
    if variable_list is None:
        variable_list = np.arange(data.shape[-1])
    data_subset = data[:, variable_list]

    # labels corresponding to each variable
    variable_names = [r"particle area [$\mu$m$^2$]",
                      r"perimeter [$\mu$m]",
                      r"circularity",
                      r"diameter [$\mu$m]",
                      r"volume [$\mu$m$^3$]"]

    variable_names_subset = np.asarray(variable_names)[variable_list]

    # create the actual figure with n histograms side by side
    plt.rcParams.update({'font.size': fontsize})
    no_of_panels = len(variable_list)
    fig, axs = plt.subplots(1, no_of_panels, squeeze=False, figsize=(no_of_panels * 4, 5))

    for i in range(no_of_panels):
        axs[0, i].hist(data_subset[:, i], bins=no_of_bins, rwidth=set_width, color=color, alpha=alpha)
        axs[0, i].set_xlabel(variable_names_subset[i])
    axs[0, 0].set_ylabel("count")

    plt.tight_layout()
    plt.show()


def scatter_hist(data, variable_list,
                 no_of_bins=20, set_width=0.9, alpha=0.8, color='darkblue', fontsize=16):
    """Creates a scatter plot with two histograms at each side of the plot.

    Takes the data containing particle areas, perimeters, circularities, equivalent sphere diameters,
    and equivalent sphere volumes, and a list of two variable indices to be used. Displays the scatter
    plot with the corresponding histograms. Different plot parameters can be passed.
    Reference: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html

    :param data: ndarray of shape (number of particles, 5)
    :param variable_list: a list of ints to create a subset from the data
    :param no_of_bins: number of histogram bins
    :param set_width: bin width
    :param alpha: transparency of the columns
    :param color: color of the columns
    :param fontsize: font size
    """

    if len(variable_list) != 2:
        print("Invalid input - please specify indices of two selected variables.")

    # select the data for plotting
    x = data[:, variable_list[0]]
    y = data[:, variable_list[1]]

    # labels corresponding to each variable
    variable_names = [r"particle area [$\mu$m$^2$]",
                      r"perimeter [$\mu$m]",
                      r"circularity",
                      r"diameter [$\mu$m]",
                      r"volume [$\mu$m$^3$]"]

    # plotting a scatter plot with two histograms
    plt.rcParams.update({'font.size': fontsize})
    fig = plt.figure(figsize=(6, 5))
    grid = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 3.5),
                            left=0.15, right=0.9, bottom=0.15, top=0.9, wspace=0.08, hspace=0.05)

    # the scatter plot
    scatter_alpha = alpha - 0.4 if alpha >= 0.5 else 0.1

    ax = fig.add_subplot(grid[1, 0])
    ax.scatter(x, y, alpha=scatter_alpha, color=color, s=15)
    ax.set_ylabel(variable_names[variable_list[1]])
    ax.set_xlabel(variable_names[variable_list[0]])

    # the two histograms
    ax_histx = fig.add_subplot(grid[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(grid[1, 1], sharey=ax)

    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax_histx.hist(x, bins=no_of_bins, rwidth=set_width, color=color, alpha=alpha)
    ax_histy.hist(y, orientation='horizontal', bins=no_of_bins, rwidth=set_width, color=color, alpha=alpha)

    plt.show()


def volume_histogram(data, d50=None, no_of_bins=20, alpha=0.8, color='darkblue', fontsize=16):
    """Displays a volume histogram for the investigated sample.

    Takes the data containing particle areas, perimeters, circularities, equivalent sphere diameters,
    and equivalent sphere volumes. Displays the volume histogram showing particle diameters in terms of
    their equivalent sphere volume. Optionally includes the characteristic d50 diameter in the plot.
    Different plot parameters can be passed.
    :param data: ndarray of shape (number of particles, 5)
    :param d50: characteristic diameter representing 50 % of particle mass
    :param no_of_bins: number of histogram bins
    :param alpha: transparency of the columns
    :param color: color of the columns
    :param fontsize: font size
    """

    # calculate a number histogram
    number_hist, bin_edges = np.histogram(data[:, 3], bins=no_of_bins)

    # determine the center values from the bin edges
    bin_centers = np.zeros(no_of_bins)
    for i in range(no_of_bins):
        bin_centers[i] = (bin_edges[i] + bin_edges[i + 1]) / 2

    # bin width
    width = (bin_edges[1] - bin_edges[0]) * 0.9

    # center volumes for each bin
    volume_centers = np.power(bin_centers / 2, 3) * 4 * np.pi / 3
    volume_sum = np.sum(data[:, -1])

    volume_bins = volume_centers * number_hist
    volume_fractions = volume_bins / volume_sum

    # plotting the volume histogram
    plt.rcParams.update({'font.size': fontsize})
    plt.figure(figsize=(6, 5))

    plt.bar(bin_centers, volume_fractions, width=width, color=color, alpha=alpha)
    if d50 is not None:
        plt.vlines(d50, 0, np.amax(volume_fractions), color='red', linestyles='dashed', linewidths=2.5)
    plt.xlabel(r"diameter [$\mu$m]")
    plt.ylabel("volume fraction")

    plt.tight_layout()
    plt.show()
