import segyio
import numpy as np
import matplotlib.pyplot as plt
# Open the SEGY file
segy_file_path = ["models/marmousi2/npy/MODEL_P-WAVE_VELOCITY_1.25m.segy", 
                  "models/marmousi2/npy/MODEL_S-WAVE_VELOCITY_1.25m.segy", 
                  "models/marmousi2/npy/MODEL_DENSITY_1.25m.segy"]
for path in segy_file_path:
    with segyio.open(path, "r", ignore_geometry=True) as segyfile:
        
        # Get the number of traces and samples per trace
        num_traces = segyfile.tracecount  # Number of traces (columns in the 2D array)
        num_samples_per_trace = segyfile.samples.size  # Number of samples per trace (rows in the 2D array)
        
        print(f"Number of traces: {num_traces}")
        print(f"Number of samples per trace: {num_samples_per_trace}")
        
        # Create an empty 2D array to hold the velocity data
        velocity_data = np.zeros((num_samples_per_trace, num_traces))

        # Read the data trace by trace
        for i in range(num_traces):
            velocity_data[:, i] = segyfile.trace[i]  # Fill each column with the trace data
        
        # Now `velocity_data` is a 2D array (num_samples_per_trace x num_traces)

    figname = path.replace('.segy', '')
    plt.figure(figsize=(10, 6))
    plt.imshow(velocity_data, cmap="jet", aspect="auto")
    plt.colorbar(label="Velocity (m/s)")
    plt.xlabel("Trace number")
    plt.ylabel("Sample index")
    plt.title(f"{figname} model")
    plt.tight_layout()
    plt.savefig(f"{figname}.png")
    plt.show()

    np.save(f'{figname}_1.25m.npy', velocity_data.astype(np.float32))
