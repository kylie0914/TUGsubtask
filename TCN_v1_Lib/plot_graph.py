import os
import matplotlib.pyplot as plt
from IPython.display import clear_output


class plotGragh:
    def __init__(self, actSplit, save ):
        self.actSplit = actSplit
        self.save = save
    
    def pltSkeleton(self, timeSec, pelvis_x, pelvis_y, pelvis_z, saveDir, saveFile,  actLabel, addFolder="raw"):
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 7))
        title = addFolder + "_" + saveFile
        
        ax[0].set_title(title)
        ax[0].set_ylabel('Side axis(X) of Pelvis [mm]')
        ax[1].set_ylabel('Vertical axis(Y) of Pelvis [mm]')
        ax[2].set_ylabel('Depth Axis of Pelvis [mm]')
        ax[2].set_xlabel('Time [Sec]')
        ax[0].plot(timeSec, pelvis_x, "r--", lw=1, label="pelvis_x")
        ax[1].plot(timeSec, pelvis_y, "g--", lw=1, label="pelvis_y")
        ax[2].plot(timeSec, pelvis_z, "b--", lw=1, label="pelvis_z")

        if self.actSplit:
            for i in range(3):
                if i ==0:
                    ax[i].axvline(x=timeSec[actLabel[0]], color='r', linestyle="--", linewidth=3, label="start move")
                    ax[i].axvline(x=timeSec[actLabel[1]], color='r', linestyle=":", linewidth=3, label="start walk")

                    ax[i].axvline(x=timeSec[actLabel[2]], color='k', linestyle="--", linewidth=3, label="start turn")
                    ax[i].axvline(x=timeSec[actLabel[3]], color='k', linestyle=":", linewidth=3, label="end turn")

                    ax[i].axvline(x=timeSec[actLabel[4]], color='g', linestyle="--", linewidth=3, label="start sit")
                    ax[i].axvline(x=timeSec[actLabel[5]], color='g', linestyle=":", linewidth=3, label="end sit")
                else:
                    ax[i].axvline(x=timeSec[actLabel[0]], color='r', linestyle="--", linewidth=3)
                    ax[i].axvline(x=timeSec[actLabel[1]], color='k', linestyle="--", linewidth=3)
                    ax[i].axvline(x=timeSec[actLabel[2]], color='g', linestyle="--", linewidth=3)

                    ax[i].axvline(x=timeSec[actLabel[3]], color='r', linestyle=":", linewidth=3)
                    ax[i].axvline(x=timeSec[actLabel[4]], color='k', linestyle=":", linewidth=3)
                    ax[i].axvline(x=timeSec[actLabel[5]], color='g', linestyle=":", linewidth=3)

        fig.legend()
        plt.show()

        if self.save:
            if  os.path.isfile(saveDir+"/"+title):
                os.remove(saveDir+"/"+title)
            fig.savefig(saveDir+"/"+title)
        plt.close()

