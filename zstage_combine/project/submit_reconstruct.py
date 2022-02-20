import os
import glob
import numpy as np
import six
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def decompress_feature(path: str) -> np.ndarray:
    for64 = {0.0: '1', -0.4: '00', 0.4: '011', -0.8: '01011', 0.8: '0100', -1.6: '010101011', 1.2: '010100', -1.2: '0101011', -2.0: '0101010101', 1.6: '01010100', -2.4: '010101010001', 2.0: '01010101001', 2.4: '0101010100001', 2.8: '01010101000001', 3.2: '010101010000000', -2.8: '010101010000001'}
    rev64 = {'1': 0.0, '00': -0.4, '011': 0.4, '01011': -0.8, '0100': 0.8, '010101011': -1.6, '010100': 1.2, '0101011': -1.2, '0101010101': -2.0, '01010100': 1.6, '010101010001': -2.4, '01010101001': 2.0, '0101010100001': 2.4, '01010101000001': 2.8, '010101010000000': 3.2, '010101010000001': -2.8}
    for128 = {0.0: '0', -0.4: '11111', -0.2: '10', 0.2: '110', -1.0: '11100001', 0.8: '1110011', -1.6: '11100000001', 0.4: '11110', -0.6: '111010', 1.2: '111001001', -0.8: '1110001', 1.4: '1110010001', 0.6: '111011', 1.0: '11100101', -1.8: '111001000001', -1.4: '1110000001', 1.6: '11100100001', -2.0: '111000000000', -1.2: '111000001', -2.4: '1110000000011010', 2.0: '1110000000010', 2.2: '111000000001100', 1.8: '111001000000', -2.2: '11100000000111', 2.4: '11100000000110111', 2.6: '111000000001101101', -2.6: '1110000000011011000', 2.8: '11100000000110110010', 3.0: '111000000001101100111', -2.8: '111000000001101100110'}
    rev128 = {'0': 0.0, '11111': -0.4, '10': -0.2, '110': 0.2, '11100001': -1.0, '1110011': 0.8, '11100000001': -1.6, '11110': 0.4, '111010': -0.6, '111001001': 1.2, '1110001': -0.8, '1110010001': 1.4, '111011': 0.6, '11100101': 1.0, '111001000001': -1.8, '1110000001': -1.4, '11100100001': 1.6, '111000000000': -2.0, '111000001': -1.2, '1110000000011010': -2.4, '1110000000010': 2.0, '111000000001100': 2.2, '111001000000': 1.8, '11100000000111': -2.2, '11100000000110111': 2.4, '111000000001101101': 2.6, '1110000000011011000': -2.6, '11100000000110110010': 2.8, '111000000001101100111': 3.0, '111000000001101100110': -2.8}
    for256 = {0.1: '011', -0.4: '11000', -0.2: '001', 0.2: '1101', 0.0: '10', -0.5: '110011', -0.1: '111', -0.3: '0101', -0.9: '110010110', 0.9: '01000110', -1.6: '010001110000', 0.5: '00000', 0.4: '01001', 1.2: '0100011111', -0.8: '11001010', 0.3: '0001', -1.1: '0100011110', 0.7: '1100100', 1.3: '0000111011', -0.7: '0100010', 0.8: '0000110', 0.6: '010000', 1.4: '01000111011', -1.0: '000011110', -0.6: '000010', -1.9: '0000111001010', 1.0: '110010111', -1.5: '010001110011', 1.1: '000011111', 1.5: '00001110001', -2.0: '01000111001001', -1.2: '0000111010', -1.4: '00001110011', 1.7: '000011100000', -1.3: '01000111010', -1.7: '000011100001', -2.5: '000011100100011010', 1.9: '00001110010111', 2.2: '0100011100100000', 1.8: '0000111001001', -2.1: '00001110010110', 1.6: '010001110001', -1.8: '0100011100101', 2.0: '00001110010000', -2.2: '010001110010001', -2.4: '00001110010001100', 2.1: '000011100100010', -2.3: '0000111001000111', 2.4: '010001110010000111', 2.5: '000011100100011011', 2.3: '01000111001000010', 2.6: '0100011100100001100', 2.8: '010001110010000110110', 3.0: '0100011100100001101110', -2.6: '0100011100100001101000', 2.9: '0100011100100001101001', 2.7: '010001110010000110101', -2.8: '0100011100100001101111'}
    rev256 = {'011': 0.1, '11000': -0.4, '001': -0.2, '1101': 0.2, '10': 0.0, '110011': -0.5, '111': -0.1, '0101': -0.3, '110010110': -0.9, '01000110': 0.9, '010001110000': -1.6, '00000': 0.5, '01001': 0.4, '0100011111': 1.2, '11001010': -0.8, '0001': 0.3, '0100011110': -1.1, '1100100': 0.7, '0000111011': 1.3, '0100010': -0.7, '0000110': 0.8, '010000': 0.6, '01000111011': 1.4, '000011110': -1.0, '000010': -0.6, '0000111001010': -1.9, '110010111': 1.0, '010001110011': -1.5, '000011111': 1.1, '00001110001': 1.5, '01000111001001': -2.0, '0000111010': -1.2, '00001110011': -1.4, '000011100000': 1.7, '01000111010': -1.3, '000011100001': -1.7, '000011100100011010': -2.5, '00001110010111': 1.9, '0100011100100000': 2.2, '0000111001001': 1.8, '00001110010110': -2.1, '010001110001': 1.6, '0100011100101': -1.8, '00001110010000': 2.0, '010001110010001': -2.2, '00001110010001100': -2.4, '000011100100010': 2.1, '0000111001000111': -2.3, '010001110010000111': 2.4, '000011100100011011': 2.5, '01000111001000010': 2.3, '0100011100100001100': 2.6, '010001110010000110110': 2.8, '0100011100100001101110': 3.0, '0100011100100001101000': -2.6, '0100011100100001101001': 2.9, '010001110010000110101': 2.7, '0100011100100001101111': -2.8}
    _pos = [0, 5, 8, 28, 35, 38, 46, 47, 48, 55, 57, 74, 78, 87, 96, 97, 102, 108, 109, 111, 112, 115, 138, 144, 146, 153, 154, 162, 165, 172, 177, 180, 181, 185, 190, 194, 196, 202, 208, 210, 215, 216, 218, 219, 224, 230, 231, 232, 233, 237, 242, 244, 245, 246, 252, 262, 265, 268, 278, 291, 299, 311, 317, 320, 330, 333, 335, 344, 349, 352, 358, 362, 364, 367, 368, 375, 379, 386, 396, 407, 411, 412, 413, 415, 416, 417, 419, 421, 424, 425, 429, 430, 431, 433, 440, 443, 446, 450, 453, 458, 459, 463, 475, 478, 486, 487, 488, 493, 496, 497, 505, 514, 519, 523, 531, 536, 539, 540, 552, 555, 564, 566, 571, 575, 579, 581, 582, 588, 590, 595, 597, 599, 600, 603, 604, 605, 607, 616, 621, 622, 639, 646, 647, 650, 653, 654, 657, 658, 664, 667, 668, 671, 672, 673, 674, 692, 694, 696, 712, 718, 731, 732, 740, 746, 747, 748, 752, 762, 766, 767, 776, 777, 778, 794, 795, 802, 806, 812, 819, 824, 829, 830, 840, 843, 847, 849, 850, 851, 855, 867, 868, 871, 874, 886, 889, 892, 894, 900, 901, 903, 907, 908, 909, 915, 916, 927, 928, 943, 944, 945, 956, 963, 964, 969, 974, 983, 985, 987, 990, 991, 995, 997, 1002, 1005, 1007, 1012, 1019, 1026, 1032, 1035, 1036, 1039, 1041, 1045, 1051, 1055, 1074, 1075, 1085, 1090, 1097, 1103, 1108, 1111, 1124, 1127, 1135, 1136, 1149, 1150, 1152, 1160, 1167, 1168, 1173, 1177, 1185, 1187, 1194, 1199, 1208, 1213, 1215, 1221, 1227, 1228, 1233, 1238, 1242, 1248, 1250, 1260, 1266, 1277, 1285, 1287, 1298, 1300, 1301, 1304, 1313, 1315, 1316, 1318, 1321, 1331, 1332, 1340, 1346, 1347, 1351, 1352, 1353, 1358, 1359, 1360, 1368, 1371, 1372, 1377, 1387, 1390, 1393, 1397, 1404, 1417, 1423, 1424, 1427, 1428, 1431, 1438, 1447, 1450, 1451, 1457, 1464, 1468, 1473, 1479, 1487, 1494, 1500, 1501, 1505, 1509, 1510, 1515, 1517, 1520, 1521, 1531, 1538, 1543, 1548, 1562, 1565, 1568, 1576, 1579, 1595, 1607, 1620, 1622, 1630, 1632, 1634, 1636, 1637, 1640, 1664, 1667, 1669, 1671, 1672, 1674, 1676, 1681, 1684, 1686, 1688, 1691, 1694, 1697, 1698, 1713, 1715, 1717, 1722, 1724, 1728, 1752, 1763, 1766, 1769, 1789, 1790, 1791, 1799, 1803, 1809, 1815, 1816, 1821, 1824, 1825, 1830, 1831, 1835, 1851, 1855, 1859, 1863, 1870, 1872, 1877, 1880, 1888, 1889, 1892, 1894, 1897, 1898, 1904, 1905, 1911, 1923, 1927, 1932, 1936, 1946, 1948, 1950, 1952, 1953, 1958, 1961, 1969, 1974, 1977, 1983, 1989, 1991, 1996, 2000, 2001, 2018, 2030, 2031, 2035, 2037, 2040, 2046]
    pos = [0, 5, 8, 28, 35, 38, 46, 47, 48, 55, 57, 74, 78, 87, 96, 97, 102, 108, 109, 111, 112, 115, 138, 144, 146, 153, 154, 158, 162, 165, 172, 177, 180, 181, 185, 190, 194, 196, 202, 208, 210, 215, 216, 218, 219, 224, 230, 231, 232, 233, 237, 242, 244, 245, 246, 252, 262, 265, 268, 278, 291, 299, 311, 312, 313, 317, 320, 330, 333, 335, 344, 349, 352, 358, 362, 364, 367, 368, 375, 379, 386, 396, 407, 411, 412, 413, 415, 416, 417, 419, 421, 424, 425, 429, 430, 431, 433, 440, 443, 446, 450, 453, 458, 459, 463, 475, 478, 486, 487, 488, 493, 496, 497, 501, 505, 514, 519, 523, 531, 536, 539, 540, 552, 555, 564, 566, 571, 575, 579, 581, 582, 588, 590, 595, 597, 599, 600, 603, 604, 605, 607, 616, 621, 622, 639, 646, 647, 650, 653, 654, 657, 658, 664, 667, 668, 671, 672, 673, 674, 692, 694, 696, 712, 718, 731, 732, 740, 746, 747, 748, 752, 762, 766, 767, 776, 777, 778, 794, 795, 802, 806, 812, 819, 824, 829, 830, 835, 840, 843, 847, 849, 850, 851, 855, 857, 864, 867, 868, 871, 874, 886, 889, 892, 894, 900, 901, 903, 907, 908, 909, 913, 915, 916, 927, 928, 943, 944, 945, 956, 961, 963, 964, 969, 974, 983, 985, 987, 990, 991, 995, 997, 1002, 1005, 1007, 1012, 1019, 1026, 1032, 1035, 1036, 1039, 1041, 1045, 1051, 1055, 1074, 1075, 1085, 1090, 1097, 1102, 1103, 1108, 1111, 1124, 1127, 1135, 1136, 1148, 1149, 1150, 1152, 1160, 1167, 1168, 1173, 1174, 1177, 1183, 1185, 1187, 1194, 1198, 1199, 1208, 1213, 1215, 1221, 1227, 1228, 1232, 1233, 1238, 1242, 1248, 1250, 1260, 1266, 1277, 1285, 1287, 1298, 1300, 1301, 1304, 1313, 1315, 1316, 1318, 1321, 1331, 1332, 1340, 1346, 1347, 1351, 1352, 1353, 1358, 1359, 1360, 1361, 1366, 1368, 1371, 1372, 1376, 1377, 1387, 1390, 1393, 1397, 1404, 1408, 1417, 1423, 1424, 1427, 1428, 1431, 1438, 1447, 1450, 1451, 1457, 1464, 1468, 1473, 1479, 1487, 1494, 1500, 1501, 1505, 1509, 1510, 1515, 1517, 1520, 1521, 1531, 1538, 1543, 1545, 1548, 1562, 1565, 1568, 1573, 1576, 1579, 1595, 1607, 1620, 1622, 1627, 1630, 1632, 1634, 1636, 1637, 1640, 1648, 1651, 1664, 1667, 1669, 1671, 1672, 1674, 1676, 1681, 1682, 1684, 1686, 1688, 1691, 1694, 1697, 1698, 1713, 1715, 1717, 1722, 1724, 1728, 1752, 1763, 1766, 1768, 1769, 1789, 1790, 1791, 1799, 1803, 1809, 1815, 1816, 1821, 1824, 1825, 1830, 1831, 1835, 1836, 1851, 1855, 1859, 1863, 1870, 1872, 1877, 1880, 1888, 1889, 1890, 1892, 1894, 1897, 1898, 1904, 1905, 1911, 1923, 1927, 1932, 1933, 1936, 1946, 1948, 1950, 1952, 1953, 1958, 1961, 1969, 1974, 1977, 1983, 1989, 1991, 1996, 2000, 2001, 2018, 2025, 2030, 2031, 2035, 2037, 2040, 2046]
    with open(path, 'rb') as f:
        feature_len = 2048
        fea = np.zeros(feature_len, dtype='<f4')
        filedata = f.read()
        filesize = f.tell()
    bytes_rate = filedata[0] * 64
    if bytes_rate == 64:
        idx = 0
        code = ''
        for x in range(1, filesize):
            #python3
            c = filedata[x]
            for i in range(8):
                if c & 128:
                    code = code + '1'
                else:
                    code = code + '0'
                c = c << 1
                if code in rev64:
                    fea[_pos[idx]] = rev64[code]
                    idx = idx + 1
                    if idx >= len(_pos):
                        return fea
                    code = ''
    elif bytes_rate == 128:
        idx = 0
        code = ''
        for x in range(1, filesize):
            #python3
            c = filedata[x]
            for i in range(8):
                if c & 128:
                    code = code + '1'
                else:
                    code = code + '0'
                c = c << 1
                if code in rev128:
                    fea[pos[idx]] = rev128[code]
                    idx = idx + 1
                    if idx > 462:
                        return fea
                    code = ''
    elif bytes_rate == 256:
        idx = 0
        code = ''
        for x in range(1, filesize):
            #python3
            c = filedata[x]
            for i in range(8):
                if c & 128:
                    code = code + '1'
                else:
                    code = code + '0'
                c = c << 1
                if code in rev256:
                    fea[pos[idx]] = rev256[code]
                    idx = idx + 1
                    if idx > 462:
                        return fea
                    code = ''
    return fea

def write_feature_file(fea: np.ndarray, path: str):
    assert fea.ndim == 1 and fea.dtype == np.float32
    with open(path, 'wb') as f:
        f.write(fea.astype('<f4').tostring())
    return True

class Decoder(nn.Module):
    def __init__(self, intermediate_dim, output_dim=463):
        super(Decoder, self).__init__()
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim

        self.relu = nn.ReLU(inplace=True)
        self.dl0 = nn.Linear(intermediate_dim, 512)
        self.dn0 = nn.BatchNorm1d(512)
        self.dl1 = nn.Linear(512, 1024)
        self.dn1 = nn.BatchNorm1d(1024)
        self.dl2 = nn.Linear(1024, 2048)
        self.dn2 = nn.BatchNorm1d(2048)
        self.dl3 = nn.Linear(2048, 4096)
        self.dn3 = nn.BatchNorm1d(4096)
        self.dl4 = nn.Linear(4096, output_dim)

    def forward(self, x):
        x = self.relu(self.dn0(self.dl0(x)))
        x = self.relu(self.dn1(self.dl1(x)))
        x = self.relu(self.dn2(self.dl2(x)))
        x = self.relu(self.dn3(self.dl3(x)))
        out = self.dl4(x)
        return out

class FeatureDataset(Dataset):
    def __init__(self, file_dir):
        self.query_fea_paths = glob.glob(os.path.join(file_dir, '*.*'))

    def __len__(self):
        return len(self.query_fea_paths)

    def __getitem__(self, index):
        vector = torch.from_numpy(np.fromfile(self.query_fea_paths[index], dtype='<f4'))
        basename = os.path.splitext(os.path.basename(self.query_fea_paths[index]))[0]
        return vector, basename

@torch.no_grad()
def reconstruct(byte_rate: str):
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(byte_rate)
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(byte_rate)
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)
    if byte_rate == '256':
        compressed_query_fea_paths = glob.glob(os.path.join(compressed_query_fea_dir, '*.*'))
        for compressed_query_fea_path in compressed_query_fea_paths:
            query_basename = get_file_basename(compressed_query_fea_path)
            reconstructed_fea = decompress_feature(compressed_query_fea_path)
            reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, query_basename + '.dat')
            write_feature_file(reconstructed_fea, reconstructed_fea_path)
    else:
        net = Decoder(int(int(byte_rate)/4))
        net.load_state_dict(torch.load(f'./Decoder_{byte_rate}.pth', map_location=torch.device('cpu')))
        net.eval()

        no_zero_dim = torch.load('./not_zero_dim.pt')

        featuredataset = FeatureDataset(compressed_query_fea_dir)
        featureloader = DataLoader(featuredataset, batch_size=8, shuffle=False, num_workers=8, pin_memory=True, drop_last=False)

        for vector, basename in featureloader:
            reconstructed = net(vector)

            expand_r = torch.zeros(reconstructed.shape[0], 2048, dtype=reconstructed.dtype)
            expand_r[:, no_zero_dim] = reconstructed

            for i, bname in enumerate(basename):
                reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir, bname + '.dat')
                with open(reconstructed_fea_path, 'wb') as bf:
                    bf.write(expand_r[i].numpy().tobytes())

    print('Decode Done' + byte_rate)
