from matplotlib import pyplot as plt
import numpy as np

figs, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)

model_no = np.argmax([np.max(history.history["val_loss"]) for history in history_before])

ax1.plot(history_before[model_no].epoch, history_before[model_no].history["loss"], label="loss")
ax1.plot(history_before[model_no].epoch, history_before[model_no].history["val_loss"], label="val_loss")
ax1.legend()
#ax1.set_ylim((0.1, 1))

ax2.plot(history_before[model_no].epoch, history_before[model_no].history["precision"], label="precision")
ax2.plot(history_before[model_no].epoch, history_before[model_no].history["val_precision"], label="val_precision")
ax2.legend()
ax2.set_ylim((0, 1))

print(history_before[model_no].history["loss"])
print(history_before[model_no].history["val_loss"])

print(history_before[model_no].history["precision"])
print(history_before[model_no].history["val_precision"])
