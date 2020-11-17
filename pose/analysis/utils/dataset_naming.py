import pandas as pd
from argparse import ArgumentParser
import numpy as np

FIX = {'Jean-Marie-Gustave-Le-Cl_zio': 'e',
       'Tomas-Transtr_mer': 'o',
       'Camilo-Jos_-Cela': 'e',
       'Frans-Eemil-Sillanp__': 'a',
       'Juan-Ram\x84n-Jim_nez': 'e',
       'Heinrich-B_ll': 'o',
       'Fr_d_ric-Mistral': 'e',
       'Jos_-Echegaray-y-Eizaguirre': 'e',
       'Andr_-Paul-Guillaume-Gide': 'e',
       'P_r-Fabian-Lagerkvist': 'a',
       'Jos_-Saramago': 'e',
       'Imre-Kert_sz': 'e'}


def main(args):
    data = pd.read_csv(args.data, encoding="ISO-8859-2", delimiter=',')
    # data = data.filter(items=['firstname', 'surname', 'category'])
    lit = data[(data['category'] == 'literature')]
    winners = lit['firstname'] + '-' + lit['surname']
    # winners = winners.replace(' ', '-')
    print(winners)

    names = np.array(winners)
    print(names)
    for a in np.nditer(names, op_flags=['readwrite'], flags=["refs_ok"]):
        name = a.item().replace(' ', '-').replace('.', ',')
        if name in FIX.keys():
            name = name.replace('_', FIX[name])
        name = name.replace('\x84', 'o')
        name = name.replace('\x8f', 'e')
        name = name.replace('\x8d', 'c')
        name = name.replace('\x88', 'a')
        if name == 'Sir-Winston-Leonard-Spencer-Churchill':
            name = 'Winston-Churchill'
        if name == 'Earl-(Bertrand-Arthur-William)-Russell':
            name = 'Bertrand-Russell'
        if name == 'Count-Maurice-(Mooris)-Polidore-Marie-Bernhard-Maeterlinck':
            name = 'Maurice-Maeterlinck'
        if name == 'Par-Fabian-Lagerkvist':
            name = 'Par-Lagerkvist'
        if name == 'BjËrnstjerne-Martinus-BjËrnson':
            name = 'Bjornsjerne-Bjornson'
        if name == 'Selma-Ottilia-Lovisa-Lagerl_f':
            name = 'Selma-Lagerlof'
        if name == 'Sir-Vidiadhar-Surajprasad-Naipaul':
            name = 'V,S,-Naiaul'
        if name == "Eugene-Gladstone-O'Neill":
            name = 'Eugene-ONeill'
        if name == 'Carl-Gustaf-Verner-von-Heidenstam':
            name = 'Verner-von-Heidenstam'
        if name == 'Mikhail-Aleksandrovich-Sholokhov':
            name = 'Mikhail-Sholokhov'
        if name == 'Ernest-Miller-Hemingway':
            name = 'Ernest-Hemingway'
        if name == 'Gerhart-Johann-Robert-Hauptmann':
            name = 'Gerhart-Hauptmann'
        if name == 'Jose-Echegaray-y-Eizaguirre':
            name = 'Jose-Echegaray'
        if name == 'Jean-Marie-Gustave-Le-Clezio':
            name = 'JMG-Le-Clezio'
        if name == 'Aleksandr-Isayevich-Solzhenitsyn':
            name = 'Aleksandr-Solzhenitsyn'
        if name == 'Christian-Matthias-Theodor-Mommsen':
            name = 'Theodor-Mommsen'
        if name == 'Carl-Friedrich-Georg-Spitteler':
            name = 'Carl-Spitteler'
        if name == 'Ivan-Alekseyevich-Bunin':
            name = 'Ivan-Bunin'
        if name == 'Halldor-Kiljan-Laxness':
            name = 'Halldor-Laxness'
        if name == 'Knut-Pedersen-Hamsun':
            name = 'Knut-Hamsun'
        if name == 'Boris-Leonidovich-Pasternak':
            name = 'Boris-Pasternak'
        if name == 'Wladyslaw-Stanislaw-Reymont':
            name = 'Wladyslaw-Reymont'
        if name == 'Andre-Paul-Guillaume-Gide':
            name = 'Andre-Gide'
        if name =='Paul-Johann-Ludwig-Heyse':
            name = 'Paul-Heyse'
        a[...] = name
        # 'Juan-Ram\x84n-Jim_nez'
        # 'Gabriel-GarcÍa-M\x88rquez'
        # Fran\x8dois - Mauriac
        # Giosu\x8f - Carducci
        print(a)
    # name_list.append(a.replace(' ', '-') for _, a in winners.items())
    # print(names)
    # for a in name_list:
    #     print(a)

    names = np.append(names, 'Bob-Dylan')
    names = np.append(names, 'Kazuo-Ishiguro')
    names = np.append(names, 'Olga-Tokarczuk')
    names = np.append(names, 'Peter-Handke')
    names = np.append(names, 'Louise-Gluck')
    print(names)
    print(names.size)

    if args.save_path != '':
        np.save(args.save_path, names)
# def parse_litterature(data):


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('--save_path', default='')
    args = parser.parse_args()
    main(args)
