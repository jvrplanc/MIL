def main():
    import musk
    import pickle

    print("Executing mi_Net with the Musk dataset")
    musk.musk_mi()
    f = open("../output/vardump", mode='rb')
    allvars = pickle.load(f)
    print(allvars)
    f.close()

if __name__ == "__main__":
    main()

