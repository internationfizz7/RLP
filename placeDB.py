from place_db.place_db_ibm import PlaceDB_ibm
from place_db.place_db_adaptec import PlaceDB_adaptec


def placeDB(args):
    benchmark = args.design_folder +'/'+ args.design + '/' + args.design
    if args.design[0:3] == "ada":
        placedb = PlaceDB_adaptec(benchmark)
    elif args.design[0:3] == "ibm":
        placedb = PlaceDB_ibm(benchmark)
    return placedb