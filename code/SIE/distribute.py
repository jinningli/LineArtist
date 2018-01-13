import os


def mkdir(path):
    if not os.path.exists(path):
        os.system('mkdir ' + path)
        print('mkdir '  + path)

def buildDirectory(rootdir):
    mkdir(os.path.join(rootdir, 'Cache'))
    mkdir(os.path.join(rootdir, 'Cache', 'SketchImage'))
    mkdir(os.path.join(rootdir, 'Cache', 'SketchImage', 'val'))
    mkdir(os.path.join(rootdir, 'Cache', 'SketchImage', 'train'))
    mkdir(os.path.join(rootdir, 'Cache', 'SketchImage', 'test'))
    mkdir(os.path.join(rootdir, 'Cache', 'ResizedImage'))
    mkdir(os.path.join(rootdir, 'Cache', 'ResizedImage', 'val'))
    mkdir(os.path.join(rootdir, 'Cache', 'ResizedImage', 'train'))
    mkdir(os.path.join(rootdir, 'Cache', 'ResizedImage', 'test'))

def copy(f, t):
    os.system("cp " + f + " " + t)
    # print("mv " + f + " " + t)


def distribute(val_weight = 0.1, test_weight = 0.2, train_weight = 0.7):

    tot = 0

    rootdir = os.getcwd()

    for root, dirs, files in os.walk("SketchImage"):
        for file in files:
            if file == ".DS_Store":
                continue
            tot += 1

    buildDirectory(rootdir)

    val_cnt = int(tot * val_weight)
    test_cnt = int(tot * test_weight)
    train_cnt = int(tot * train_weight)

    cnt = 0

    for root, dirs, files in os.walk("SketchImage"):
        for file in files:
            if file == ".DS_Store":
                continue
            if cnt <= val_cnt:
                copy(os.path.join(root, file), os.path.join(rootdir, 'Cache', 'SketchImage', 'val', file))
            else:
                if cnt <= val_cnt + test_cnt:
                    copy(os.path.join(root, file), os.path.join(rootdir, 'Cache', 'SketchImage', 'test', file))
                else:
                    if cnt <= val_cnt + test_cnt + train_cnt:
                        copy(os.path.join(root, file), os.path.join(rootdir, 'Cache', 'SketchImage', 'train', file))
            cnt += 1
            if cnt % 1000 == 0:
                print("Copy " + str(cnt) + " of " + str(tot))


    cnt = 0

    for root, dirs, files in os.walk("ResizedImage"):
        for file in files:
            if file == ".DS_Store":
                continue
            if cnt <= val_cnt:
                copy(os.path.join(root, file), os.path.join(rootdir, 'Cache', 'ResizedImage', 'val', file))
            else:
                if cnt <= val_cnt + test_cnt:
                    copy(os.path.join(root, file), os.path.join(rootdir, 'Cache', 'ResizedImage', 'test', file))
                else:
                    if cnt <= val_cnt + test_cnt + train_cnt:
                        copy(os.path.join(root, file), os.path.join(rootdir, 'Cache', 'ResizedImage', 'train', file))
            cnt += 1
            if cnt % 1000 == 0:
                print("Copy " + str(cnt) + " of " + str(tot))