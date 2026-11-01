"""Run the two Oct 10 image probes without loading the same model twice."""
from run_image_audio_probe import main, ROOT

runtime = None
for image, directory in [
    ('20230713_145512_23239672_w1920.webp', 'image_probe_station_greedy_20261010'),
    ('960.png', 'image_probe_trains_greedy_20261010'),
]:
    runtime = main([
        'infer', '--image', str(ROOT / image), '--output-dir', str(ROOT / 'eval_results' / directory),
        '--device', 'cuda:0', '--split-turns',
    ], runtime=runtime)
