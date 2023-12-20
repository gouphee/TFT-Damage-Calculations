import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def damage_calculation(hp, armor, mr):

    warmogPhy = (hp + 600)*1.08 * ((100+armor)/100)
    warmogMag = (hp + 600)*1.08 * ((100+mr)/100)

    warmog = (hp + 600)*1.08 * ((100+mr)/100)

    steadfastPhy1 = (hp + 250) * (100+armor+20)/100 * (1/.85)
    steadfastPhy2 =  (hp + 250)  * (100+armor+20)/100 * (1/.92)
    steadfastMag1 = (hp + 250)  * (100+mr)/100 * (1/.85)
    steadfastMag2 =  (hp + 250)  * (100+mr)/100 * (1/.92)

    # # average when you just average the two DR values
    # steadfastPhy = 2*(steadfastPhy1*steadfastPhy2) / (steadfastPhy1 + steadfastPhy2)
    # steadfastMag = 2*(steadfastMag1*steadfastMag2) / (steadfastMag1 + steadfastMag2)

    # average when you split the hp pool
    steadfastPhy = (steadfastPhy1 + steadfastPhy2) / 2
    steadfastMag = (steadfastMag1 + steadfastMag2) / 2

    return [warmog, steadfastPhy, steadfastMag]

def resistance_graph():
    resistance = [[],[],[], range(201)]
    for i in range(201):
        for index, val in enumerate(damage_calculation(1800, i, i)):
            resistance[index].append(val)
    df = pd.DataFrame(data=resistance)
    df = df.transpose()
    df.columns = ['warmog', 'steadfastPhy', 'steadfastMag', 'Resistance']
    df['PhyRatio'] = (df['steadfastPhy']/df['warmog'])
    df['MagRatio'] = (df['steadfastMag']/df['warmog'])
    df['PhyDiff'] = (df['steadfastPhy'] - df['warmog'])
    df['MagDiff'] = (df['steadfastMag'] - df['warmog'])
    diff = df[['Resistance', 'PhyDiff', 'MagDiff']]

    sns.set_theme(style="darkgrid")

    g = sns.lineplot(data=pd.melt(diff, ['Resistance']), x='Resistance', y="value", hue='variable')

    g.set(ylabel='Effective HP Difference (Steadfast - Warmog)')

    plt.show()

def hp_graph():
    hp = [[],[],[], range(10001)]
    for i in range(10001):
        for index, val in enumerate(damage_calculation(i, 50, 50)):
            hp[index].append(val)
    df = pd.DataFrame(data=hp)
    df = df.transpose()
    df.columns = ['warmog', 'steadfastPhy', 'steadfastMag', 'HP']
    df['PhyRatio'] = (df['steadfastPhy']/df['warmog'])
    df['MagRatio'] = (df['steadfastMag']/df['warmog'])
    df['PhyDiff'] = (df['steadfastPhy'] - df['warmog'])
    df['MagDiff'] = (df['steadfastMag'] - df['warmog'])
    diff = df[['HP', 'PhyDiff', 'MagDiff']]

    sns.set_theme(style="darkgrid")

    g = sns.lineplot(data=pd.melt(diff, ['HP']), x='HP', y="value", hue='variable')

    g.set(ylabel='Effective HP Difference (Steadfast - Warmog)')

    plt.show()

def surface_plot():
    arrayP = []
    arrayM = []
    for i in range(10001):
        resistP = []
        resistM = []
        for j in range(201):
            vals = damage_calculation(i, j, j)
            resistP.append(vals[1]-vals[0])
            resistM.append(vals[2]-vals[0])
        arrayP.append(resistP)
        arrayM.append(resistM)
    dfP = pd.DataFrame(arrayP)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create an X-Y mesh of the shape of the dataframe
    X, Y = np.meshgrid(range(len(dfP.columns)), range(len(dfP.index)))
    
    # Convert the dataframe to a 2D array for plotting
    Z = dfP.values

    z_min, z_max = Z.min(), Z.max()
    z_range = z_max - z_min
    ax.set_zlim(-z_max - z_range * 0.1, z_max + z_range * 0.1)

    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, vmin=-z_max, vmax=z_max, linewidth=0)

    ax.set_xlabel("Resistances")
    ax.set_ylabel("HP")
    ax.set_zlabel("Effective HP Difference (Steadfast - Warmog)")

    # Add a color bar which maps values to colors
    fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)

    plt.show()


def main():
    # resistance_graph()
    # hp_graph()
    # surface_plot()

    base = damage_calculation(100, 0, 0)

    print(base)
    


if __name__ == "__main__":
    main()
