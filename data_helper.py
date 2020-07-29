import csv
import numpy as np

from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class DataHelper():
    def __init__(self, sequence_max_length=1024):
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’"/|_#$%ˆ&*˜‘+=<>()[]{} '
        self.char_dict = {}
        self.UNK = 68
        self.sequence_max_length = sequence_max_length
        for i,c in enumerate(self.alphabet):
            self.char_dict[c] = i+1

    def char2vec(self, text):
        data = np.zeros(self.sequence_max_length)
        for i in range(0, len(text)):
            if i >= self.sequence_max_length:
                return data
            elif text[i] in self.char_dict:
                data[i] = self.char_dict[text[i]]
            else:
                data[i] = self.UNK
        return np.array(data)

    def load_csv_file(self, filename, num_classes, train=True, one_hot=False):
        if train:
            s1 = 120000
        else:
            s1 = 7600
        all_data =np.zeros(shape=(s1, self.sequence_max_length), dtype=np.int)
        labels =np.zeros(shape=(s1, 1), dtype=np.int)

        # labels = []
        with open(filename) as f:
            reader = csv.DictReader(f, fieldnames=['class'], restkey='fields')
            # reader = np.genfromtxt(f)
            for i,row in enumerate(reader):
                if one_hot:
                    one_hot = np.zeros(num_classes)
                    one_hot[int(row['class']) - 1] = 1
                    labels[i] = one_hot
                else:
                    labels[i] = int(row['class']) - 1
                text = row['fields'][-1].lower()
                all_data[i] = self.char2vec(text)
        f.close()
        return all_data, labels

    def load_dataset(self, dataset_path):
        with open(dataset_path+"classes.txt") as f:
            classes = []
            for line in f:
                classes.append(line.strip())
        f.close()
        num_classes = len(classes)
        train_data, train_label = self.load_csv_file(dataset_path + 'train.csv', num_classes)
        test_data, test_label = self.load_csv_file(dataset_path + 'test.csv', num_classes, train=False)
        print(train_data.shape, test_data.shape)
        return train_data, train_label, test_data, test_label

    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
        # for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            batch = shuffled_data[start_index:end_index]
            # batch_data, label = batch[:,  self.sequence_max_length-1], batch[:, -1]
            batch_data, label = np.split(batch, [self.sequence_max_length],axis=1)
            yield np.array(batch_data, dtype=np.int), label


def get_mnist_loaders(batch_size=128, test_batch_size=1000, perc=1.0):
    transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


def get_cifar_loaders(batch_size=128, test_batch_size=1000):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    train_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar', train=True, download=True, transform=transform_train), batch_size=batch_size,
        shuffle=True, num_workers=3)

    test_loader = DataLoader(
        datasets.CIFAR10(root='.data/cifar', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=3)

    return train_loader, test_loader, None

classes={'adorable_face': 0,
 'amazing_cars': 1,
 'attractive_lady': 2,
 'awesome_animals': 3,
 'awesome_nature': 4,
 'awesome_scene': 5,
 'bad_graffiti': 6,
 'bright_lights': 7,
 'bright_rainbow': 8,
 'busy_city': 9,
 'calm_ocean': 10,
 'calm_street': 11,
 'candid_girls': 12,
 'charming_smile': 13,
 'christian_festival': 14,
 'classic_architecture': 15,
 'clean_water': 16,
 'colorful_autumn': 17,
 'crazy_rain': 18,
 'crazy_storm': 19,
 'crowded_city': 20,
 'crying_child': 21,
 'cute_face': 22,
 'dark_shadows': 23,
 'dark_tower': 24,
 'dead_end': 25,
 'deadly_fire': 26,
 'dry_winter': 27,
 'dusty_glasses': 28,
 'empty_room': 29,
 'empty_street': 30,
 'evil_dog': 31,
 'evil_robot': 32,
 'excited_girls': 33,
 'fantastic_city': 34,
 'fluffy_cat': 35,
 'fluffy_puppy': 36,
 'fresh_flowers': 37,
 'friendly_dog': 38,
 'friendly_smile': 39,
 'frightened_child': 40,
 'golden_dragon': 41,
 'gorgeous_dress': 42,
 'great_reflection': 43,
 'great_sky': 44,
 'great_street': 45,
 'happy_baby': 46,
 'happy_christmas': 47,
 'happy_dog': 48,
 'hardcore_band': 49,
 'hardcore_terror': 50,
 'healthy_teeth': 51,
 'heavy_storm': 52,
 'holy_mountains': 53,
 'horrible_monster': 54,
 'hot_pot': 55,
 'incredible_city': 56,
 'innocent_eyes': 57,
 'inspirational_bible': 58,
 'laughing_children': 59,
 'lazy_morning': 60,
 'lonely_island': 61,
 'lost_dog': 62,
 'lost_weight': 63,
 'lovely_autumn': 64,
 'misty_morning': 65,
 'nasty_spider': 66,
 'noisy_bird': 67,
 'playful_cats': 68,
 'poor_cat': 69,
 'powerful_waves': 70,
 'proud_student': 71,
 'rainy_forest': 72,
 'relaxing_beach': 73,
 'relaxing_evening': 74,
 'relaxing_hotel': 75,
 'rotten_apple': 76,
 'scary_eyes': 77,
 'scary_zombie': 78,
 'screaming_baby': 79,
 'screaming_face': 80,
 'sexy_blonde': 81,
 'silly_kids': 82,
 'silly_toys': 83,
 'slippery_snow': 84,
 'smooth_curves': 85,
 'strange_building': 86,
 'stunning_sunset': 87,
 'stupid_pet': 88,
 'super_star': 89,
 'tasty_chocolate': 90,
 'tiny_dog': 91,
 'traditional_farm': 92,
 'traveling_magazine': 93,
 'ugly_fly': 94,
 'ugly_wall': 95,
 'weird_alien': 96,
 'wet_cat': 97,
 'wild_hair': 98,
 'yummy_food': 99}

anp_classes=dict((v,k) for k,v in classes.items())

cls=anp_classes.values()
a=set([x.split('_')[0] for x in cls])
n=set([x.split('_')[1] for x in cls])

adj_classes=dict([(v,i) for i,v in enumerate(a)])
noun_classes=dict([(v,i) for i,v in enumerate(n)])

def get_galaxyZoo_loaders(batch_size=20, test_batch_size=20):
    from torch.utils.data.sampler import SubsetRandomSampler
    # batch_size=training_config['batch_size']
    # test_batch_size=training_config['test_batch_size']
    # size=training_config['img_size']
    # crop_size=training_config['crop_size']
    # transform_train = transforms.Compose([
    #         transforms.Grayscale(num_output_channels=1),
    #         transforms.CenterCrop((crop_size,crop_size)),
    #         transforms.Resize(size),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (0.5,)),
    #     ])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    def target_transform(x):
        anp=anp_classes[x]
        adj,noun=anp.split('_')
        return classes[anp],adj_classes[adj],noun_classes[noun]

    # transform_test = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=1),
    #     # transforms.CenterCrop((64,64)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,)),
    # ])

    # gz_root = '/home/cs19mtech11019/cs19mtech11024/imageFolder'
    # gz_root = '/content/drive/My Drive/imageFolder'
    gz_root = '/mnt/f/IITH/research/physics/galaxy_zoo/GalaxyClassification/imageFolder_small'
    gz_root = '/mnt/f/IITH/research/cs/mtvso_task/dataset'
    # gz_root = '/home/nilesh/raghav/mtvso_task/dataset'

    gz_dataset = datasets.ImageFolder(root=gz_root
            # ,train=True, download=True
            , transform=transform_train,
        target_transform=target_transform
        )

    # total_images = len(gz_dataset)

    # train_dataset, test_dataset = random_split(gz_dataset,[
    #     int(0.9*total_images),
    #     int(0.1*total_images)
    # ])

    split = .9
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(gz_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[:split], indices[split:]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)



    train_loader = DataLoader(gz_dataset
        ,batch_size=batch_size,
        shuffle=False, num_workers=1, drop_last=True
        ,sampler=train_sampler
    )

    # train_eval_loader = DataLoader(validation_dataset
    #     ,batch_size=test_batch_size, shuffle=True, num_workers=2, drop_last=True
    # )

    test_loader = DataLoader(gz_dataset
        ,batch_size=test_batch_size,
        shuffle=False, num_workers=1, drop_last=True
        ,sampler=test_sampler
    )

    return train_loader, test_loader, gz_dataset


if __name__ == '__main__':
    sequence_max_length = 1014
    batch_size = 32
    num_epochs = 32
    database_path = '.data/ag_news/'
    data_helper = DataHelper(sequence_max_length=sequence_max_length)
    train_data, train_label, test_data, test_label = data_helper.load_dataset(database_path)
    train_batches = data_helper.batch_iter(np.column_stack((train_data, train_label)), batch_size, num_epochs)
    for batch in train_batches:
        train_data_b,label = batch
        break
