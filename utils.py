import resource
import types

import keras
import keras.backend as K
from keras.models import model_from_json
from pympler import muppy, summary, tracker
from pympler.web import start_in_background

from model import clipped_relu, selu


# these text/int characters are modified
# from the DS2 github.com/baidu-research/ba-dls-deepspeech


def save_trimmed_model(model, name):
    jsonfilename = str(name) + ".json"
    weightsfilename = str(name) + ".h5"

    # # serialize model to JSON
    with open(jsonfilename, "w") as json_file:
        json_file.write(model.to_json())

    # # serialize weights to HDF5
    model.save_weights(weightsfilename)

    return


def save_model(model, name):
    if name:
        jsonfilename = str(name) + "/model.json"
        weightsfilename = str(name) + "/model.h5"

        # # serialize model to JSON
        with open(jsonfilename, "w") as json_file:
            json_file.write(model.to_json())

        print("Saving model at:", jsonfilename, weightsfilename)
        model.save_weights(weightsfilename)

        # save model as combined in single file - contrains arch/weights/config/state
        model.save(str(name) + "/cmodel.h5")

    return


def load_model_checkpoint(root_path):
    # this is a terrible hack
    from keras.utils.generic_utils import get_custom_objects
    get_custom_objects().update({"clipped_relu": clipped_relu})
    get_custom_objects().update({"selu": selu})

    model_json = root_path + "model.json"  # architecture
    model_weights = root_path + "model.h5"

    with open(model_json, 'r') as json_file:
        loaded_model_json = json_file.read()

        K.set_learning_phase(1)
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(model_weights)

    return loaded_model


memlist = []


class MemoryCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log={}):
        x = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        web_browser_debug = True
        print(x)

        if x > 40000:
            if web_browser_debug:
                if epoch == 0:
                    start_in_background()
                    tr = tracker.SummaryTracker()
                    tr.print_diff()
            else:
                global memlist
                all_objects = muppy.get_objects(include_frames=True)
                # print(len(all_objects))
                sum1 = summary.summarize(all_objects)
                memlist.append(sum1)
                summary.print_(sum1)
                if len(memlist) > 1:
                    # compare with last - prints the difference per epoch
                    diff = summary.get_diff(memlist[-2], memlist[-1])
                    summary.print_(diff)
                my_types = muppy.filter(all_objects, Type=types.ClassType)

                for t in my_types:
                    print(t)

    #########################################################
