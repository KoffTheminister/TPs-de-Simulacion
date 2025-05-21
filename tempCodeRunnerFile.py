    x = np.linspace(media - 4*desviacion, media + 4*desviacion, 1000)
    y = norm.pdf(x, media, desviacion)
    y *= (bin_edges[1] - bin_edges[0]) #
    plt.figure(figsize=(10, 4))
    plt.plot(x, y, label=f'N({media}, {desviacion}²)', color='blue')