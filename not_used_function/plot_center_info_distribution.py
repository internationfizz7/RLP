### data is in the data_processed format

centers = []
    
for i in range(len(data)-1):
    feature = data[i][1]
    centers.append(feature[1])

for node_name in data[0][0][0]:
    center_distance_list = []
    for i in range(len(centers)):
        center_distance_list.append(centers[i][node_name])
    counts, bin_edges = np.histogram(center_distance_list, bins='auto')
    plt.bar(range(len(counts)), counts, tick_label=[f"{int(bin_edges[i])}-{int(bin_edges[i+1])}" for i in range(len(counts))])
    plt.xlabel("Distance Ranges")
    plt.ylabel("Frequency")
    plt.title('center_distance')
    plt.savefig(args.plt_path +"center_info/"+ node_name + '.png', format="png", dpi = 300)
    plt.cla()