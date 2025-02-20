regions = []
    
for i in range(len(data)-1):
    feature = data[i][1]
    regions.append(feature[2])

for node_name in data[0][0][0]:
    coor_list = []
    for i in range(len(regions)):
        coor_list.append((regions[i][node_name][1], regions[i][node_name][3]))
    x_coords = [point[0] for point in coor_list]
    y_coords = [point[1] for point in coor_list]
    plt.scatter(x_coords, y_coords, c='blue', marker='o', s=50, alpha=0.7)
    plt.xlabel("X coor")
    plt.ylabel("Y coor")
    plt.title('region info')
    plt.grid(True)

    plt.savefig(args.plt_path +"region_info/"+ node_name + '.png', format="png", dpi = 300)
    plt.cla()