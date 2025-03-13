import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

class ECALDataset:
    def __init__(self, N, csv_file="ecal_hits_table.csv", min_particles=2, max_particles=2, variance=0,
                 random_sector=False, force_same_sector=False, K=20):
        """
        Initializes the dataset.
        
        Parameters:
            N (int): Number of events.
            csv_file (str): Path to the CSV file containing the ECal hits.
            min_particles (int): Minimum number of particles per event.
            max_particles (int): Maximum number of particles per event.
            variance (int): Number of adjacent flashes to attempt (if nonzero, extra hits are added).
            random_sector (bool): If True, choose a random sector (1-6) for each particle;
                                  if False, use only sector 1.
            force_same_sector (bool): If True and if an event has >=2 particles,
                                      force at least one pair of particles to be in the same sector.
                                      For example, if min=max=2, both particles are in the same sector.
                                      If min=max=3, two particles share one sector and the third is in a different sector.
            K (int): Maximum number of hits per event.
        """
        self.N = N
        self.csv_file = csv_file
        self.min_particles = min_particles
        self.max_particles = max_particles
        self.variance = variance
        self.random_sector = random_sector
        self.force_same_sector = force_same_sector
        self.K = K
        # Preallocate the dataset arrays:
        # X: (N, K, 8) initialized to zeros.
        self.X = np.zeros((N, self.K, 8), dtype=np.float32)
        # y: (N, K, 1) initialized to -1.
        self.y = -1 * np.ones((N, self.K, 1), dtype=np.int32)
        
        self._load_csv()
        self._create_dataset()

    def _load_csv(self):
        """Loads the CSV file and organizes hits by sector and layer."""
        self.hits_df = pd.read_csv(self.csv_file)
        # Build a dictionary for sectors 1-6 and layers 1, 2, and 3.
        self.df_sector_layer = {}
        for sector in range(1, 7):
            for layer in [1, 2, 3]:
                self.df_sector_layer[(sector, layer)] = self.hits_df[
                    (self.hits_df["Sector"] == sector) & (self.hits_df["Layer"] == layer)
                ].reset_index(drop=True)

    @staticmethod
    def _orientation(p, q, r):
        """Return orientation of triplet (p, q, r).
        0 -> collinear, 1 -> clockwise, 2 -> counterclockwise."""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if np.abs(val) < 1e-6:
            return 0
        return 1 if val > 0 else 2

    @staticmethod
    def _on_segment(p, q, r):
        """Return True if point q lies on segment pr (assuming p, q, r are collinear)."""
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
            return True
        return False

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4):
        """Return True if segments p1p2 and p3p4 intersect."""
        o1 = ECALDataset._orientation(p1, p2, p3)
        o2 = ECALDataset._orientation(p1, p2, p4)
        o3 = ECALDataset._orientation(p3, p4, p1)
        o4 = ECALDataset._orientation(p3, p4, p2)
        
        # General case.
        if o1 != o2 and o3 != o4:
            return True
        
        # Special Cases.
        if o1 == 0 and ECALDataset._on_segment(p1, p3, p2):
            return True
        if o2 == 0 and ECALDataset._on_segment(p1, p4, p2):
            return True
        if o3 == 0 and ECALDataset._on_segment(p3, p1, p4):
            return True
        if o4 == 0 and ECALDataset._on_segment(p3, p2, p4):
            return True
        
        return False

    @staticmethod
    def _compute_intersection(p1, p2, p3, p4):
        """
        Compute the intersection point of the lines (not segments) defined by p1p2 and p3p4.
        Returns a tuple (x, y) if the lines are not parallel; otherwise, returns None.
        """
        x1, y1 = p1; x2, y2 = p2
        x3, y3 = p3; x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if np.abs(denom) < 1e-6:
            return None
        num_x = (x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)
        num_y = (x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)
        return (num_x/denom, num_y/denom)
        
    def _flash_adjacent_hits_local(self, event_X, event_y, hit_index, df_layer, original_row, global_particle_id):
        """
        Locally flash adjacent hits by sampling additional candidates and writing into the local
        event buffers.
        """
        max_hits = event_X.shape[0]
        num_flashes = np.random.randint(0, self.variance + 1)
        for _ in range(num_flashes):
            if hit_index >= max_hits:
                break
            delta = np.random.choice([-1, 1])
            new_comp = original_row["Component"] + delta
            possible = df_layer[df_layer["Component"] == new_comp]
            if not possible.empty:
                flash_row = possible.sample(n=1).iloc[0]
                event_X[hit_index, 0] = flash_row["xo"]
                event_X[hit_index, 1] = flash_row["yo"]
                event_X[hit_index, 2] = flash_row["zo"]
                event_X[hit_index, 3] = flash_row["xe"]
                event_X[hit_index, 4] = flash_row["ye"]
                event_X[hit_index, 5] = flash_row["ze"]
                event_X[hit_index, 6] = flash_row["Component"]
                event_X[hit_index, 7] = flash_row["Layer"]
                event_y[hit_index, 0] = global_particle_id
                hit_index += 1
        return hit_index

    def _create_dataset(self):
        """
        Creates the dataset by sampling hits and (if variance > 0) adding extra flashed hits.
        If any layer fails (no candidate found within the maximum trials), the event is discarded
        and rebuilt.
        """
        global_particle_id = 0  # Unique id for each particle across all events.
        max_hits = self.K
        
        for event in tqdm(range(self.N)):
            while True:  # Retry loop for the event.
                event_failed = False
                # Create temporary buffers for the event.
                event_X = np.zeros((max_hits, 8), dtype=np.float32)
                event_y = -1 * np.ones((max_hits, 1), dtype=np.int32)
                local_hit_index = 0
                event_start_global_particle_id = global_particle_id

                n_particles = np.random.randint(self.min_particles, self.max_particles + 1)
                
                # Determine sectors for particles in this event.
                particle_sectors = []
                if self.force_same_sector and n_particles >= 2:
                    common_sector = np.random.randint(1, 7) if self.random_sector else 1
                    particle_sectors.extend([common_sector, common_sector])
                    for _ in range(n_particles - 2):
                        if self.random_sector:
                            possible = [s for s in range(1, 7) if s != common_sector]
                            other_sector = np.random.choice(possible)
                        else:
                            other_sector = 1
                        particle_sectors.append(other_sector)
                else:
                    for _ in range(n_particles):
                        sector = np.random.randint(1, 7) if self.random_sector else 1
                        particle_sectors.append(sector)
                
                # Process each particle.
                for sector in particle_sectors:
                    if local_hit_index >= max_hits:
                        break
                    
                    # --- Layer 1 ---
                    try:
                        row1 = self.df_sector_layer[(sector, 1)].sample(n=1).iloc[0]
                    except Exception as e:
                        print("Layer 1 sampling error for sector", sector, "in event", event)
                        event_failed = True
                        break
                    event_X[local_hit_index, 0] = row1["xo"]
                    event_X[local_hit_index, 1] = row1["yo"]
                    event_X[local_hit_index, 2] = row1["zo"]
                    event_X[local_hit_index, 3] = row1["xe"]
                    event_X[local_hit_index, 4] = row1["ye"]
                    event_X[local_hit_index, 5] = row1["ze"]
                    event_X[local_hit_index, 6] = row1["Component"]
                    event_X[local_hit_index, 7] = row1["Layer"]
                    event_y[local_hit_index, 0] = global_particle_id
                    local_hit_index += 1
                    
                    row_layer1 = row1.copy()
                    p1 = (row1["xo"], row1["yo"])
                    p2 = (row1["xe"], row1["ye"])
                    
                    # --- Layer 2 ---
                    if local_hit_index >= max_hits:
                        break
                    max_trials = 1000
                    trial = 0
                    found = False
                    candidate = None
                    while trial < max_trials and not found:
                        candidate = self.df_sector_layer[(sector, 2)].sample(n=1).iloc[0]
                        p3 = (candidate["xo"], candidate["yo"])
                        p4 = (candidate["xe"], candidate["ye"])
                        if ECALDataset._segments_intersect(p1, p2, p3, p4):
                            found = True
                        trial += 1
                    if not found:
                        print("Layer 2 fail for sector", sector, "in event", event)
                        event_failed = True
                        break
                    event_X[local_hit_index, 0] = candidate["xo"]
                    event_X[local_hit_index, 1] = candidate["yo"]
                    event_X[local_hit_index, 2] = candidate["zo"]
                    event_X[local_hit_index, 3] = candidate["xe"]
                    event_X[local_hit_index, 4] = candidate["ye"]
                    event_X[local_hit_index, 5] = candidate["ze"]
                    event_X[local_hit_index, 6] = candidate["Component"]
                    event_X[local_hit_index, 7] = candidate["Layer"]
                    event_y[local_hit_index, 0] = global_particle_id
                    local_hit_index += 1
                    
                    row_layer2 = candidate.copy()
                    p_layer2_1 = (row_layer2["xo"], row_layer2["yo"])
                    p_layer2_2 = (row_layer2["xe"], row_layer2["ye"])
                    
                    # --- Layer 3 ---
                    if local_hit_index >= max_hits:
                        break
                    max_trials = 1000
                    trial = 0
                    found = False
                    candidate = None
                    while trial < max_trials and not found:
                        candidate = self.df_sector_layer[(sector, 3)].sample(n=1).iloc[0]
                        p5 = (candidate["xo"], candidate["yo"])
                        p6 = (candidate["xe"], candidate["ye"])
                        inter1 = ECALDataset._compute_intersection(p1, p2, p5, p6)
                        inter2 = ECALDataset._compute_intersection(p_layer2_1, p_layer2_2, p5, p6)
                        if inter1 is not None and inter2 is not None:
                            dist = np.sqrt((inter1[0] - inter2[0])**2 + (inter1[1] - inter2[1])**2)
                            if dist < 0.005:
                                found = True
                        trial += 1
                    if not found:
                        print("Layer 3 fail for sector", sector, "in event", event)
                        event_failed = True
                        break
                    event_X[local_hit_index, 0] = candidate["xo"]
                    event_X[local_hit_index, 1] = candidate["yo"]
                    event_X[local_hit_index, 2] = candidate["zo"]
                    event_X[local_hit_index, 3] = candidate["xe"]
                    event_X[local_hit_index, 4] = candidate["ye"]
                    event_X[local_hit_index, 5] = candidate["ze"]
                    event_X[local_hit_index, 6] = candidate["Component"]
                    event_X[local_hit_index, 7] = candidate["Layer"]
                    event_y[local_hit_index, 0] = global_particle_id
                    local_hit_index += 1
                    
                    row_layer3 = candidate.copy()
                    
                    # --- Variance Flashing (if enabled) ---
                    if self.variance > 0:
                        local_hit_index = self._flash_adjacent_hits_local(event_X, event_y, local_hit_index,
                                                                          self.df_sector_layer[(sector, 1)],
                                                                          row_layer1, global_particle_id)
                        if local_hit_index >= max_hits:
                            break
                        local_hit_index = self._flash_adjacent_hits_local(event_X, event_y, local_hit_index,
                                                                          self.df_sector_layer[(sector, 2)],
                                                                          row_layer2, global_particle_id)
                        if local_hit_index >= max_hits:
                            break
                        local_hit_index = self._flash_adjacent_hits_local(event_X, event_y, local_hit_index,
                                                                          self.df_sector_layer[(sector, 3)],
                                                                          row_layer3, global_particle_id)
                        if local_hit_index >= max_hits:
                            break
                    
                    global_particle_id += 1
                # End of particle loop.
                if event_failed:
                    # If a layer failed, revert any particle id changes and retry the event.
                    global_particle_id = event_start_global_particle_id
                    print("Retrying event", event)
                    continue  # Retry the while-loop for this event.
                else:
                    # Commit the local event buffers into the final dataset.
                    self.X[event, :local_hit_index, :] = event_X[:local_hit_index, :]
                    self.y[event, :local_hit_index, :] = event_y[:local_hit_index, :]
                    break  # Move to the next event.

    def get_data(self):
        """Returns the dataset arrays (X and y)."""
        return self.X, self.y