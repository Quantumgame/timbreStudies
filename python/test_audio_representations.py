import matplotlib.pylab as plt
from lib import utils
from lib import auditory
from lib import fourier

def run_test():
    audio, fs = utils.audio_data("/Users/baptistecaramiaux/Work/Projects/TimbreProject_Thoret/Code and data/timbreStudies/ext/sounds/Iverson1993Whole/01.W.Violin.aiff")
    # rep = fourier.spectrum(audio, fs)
    # plt.plot(rep)
    rep = fourier.strf(audio, fs)
    print(rep.shape)
    plt.imshow(rep)
    plt.show()

if __name__=="__main__":
    run_test()