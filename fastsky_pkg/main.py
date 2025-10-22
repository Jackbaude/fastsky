#!/usr/bin/env python3
"""Real-time satellite tracker CLI using dsgp4 GPU batch propagation."""

import argparse
from dsgp4.tle import load
from dsgp4 import initialize_tle
from dsgp4.util import from_datetime_to_mjd, gstime, propagate_batch
import torch
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import time

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.console import Console
from rich.text import Text
from rich.columns import Columns


class Observer:
    """Observer location on Earth."""
    
    def __init__(self, latitude, longitude, altitude=0.0):
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.lat_rad = np.radians(latitude)
        self.lon_rad = np.radians(longitude)


def geodetic_to_ecef(lat_rad, lon_rad, alt_km):
    """Convert geodetic to ECEF coordinates."""
    a = 6378.137
    f = 1.0 / 298.257223563
    e2 = 2 * f - f * f
    
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    N = a / np.sqrt(1 - e2 * sin_lat * sin_lat)
    
    x = (N + alt_km) * cos_lat * np.cos(lon_rad)
    y = (N + alt_km) * cos_lat * np.sin(lon_rad)
    z = (N * (1 - e2) + alt_km) * sin_lat
    
    return np.array([x, y, z])


def make_display(tles, tle_batch, observer, iteration, target_time, 
                 num_visible, num_high, num_medium, num_low,
                 sorted_indices, elevations, azimuths, ranges,
                 prop_time, topo_time, total_time, update_interval):
    """Create the display layout."""
    
    layout = Layout()
    
    layout.split_column(
        Layout(name="header", size=7),
        Layout(name="stats", size=8),
        Layout(name="satellites"),
        Layout(name="footer", size=3)
    )
    
    title = Text(f"FastSky - Real-Time Satellite Tracker", style="bold white on blue")
    subtitle = f"Update #{iteration} | {target_time.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    location = f"📍 {observer.latitude:.4f}°N, {observer.longitude:.4f}°W | 🛰️  {len(tles)} satellites"
    
    header = Panel(
        f"[bold cyan]{title}[/bold cyan]\n{subtitle}\n{location}",
        style="blue"
    )
    layout["header"].update(header)
    
    stats_text = f"""[bold yellow]VISIBILITY SUMMARY[/bold yellow]
[green]Total above horizon:[/green]     {num_visible:>4} satellites
[cyan]High passes (>20°):[/cyan]      {num_high:>4} satellites  
[yellow]Medium passes (10-20°):[/yellow] {num_medium:>4} satellites
[dim]Low passes (<10°):[/dim]       {num_low:>4} satellites

[magenta]⚡ GPU propagation:[/magenta]      {prop_time:.3f}s ({len(tles)/prop_time:,.0f} sats/sec)
[blue]🔄 Topocentric calc:[/blue]      {topo_time:.3f}s
[bold green]📊 Total:[/bold green]               {total_time:.3f}s ({len(tles)/total_time:,.0f} sats/sec)"""
    
    stats_panel = Panel(stats_text, title="Statistics", border_style="green")
    layout["stats"].update(stats_panel)
    
    # Create multiple tables to show all satellites
    tables = []
    sats_per_table = 25
    num_tables = min(4, (len(sorted_indices) + sats_per_table - 1) // sats_per_table)
    
    for table_idx in range(num_tables):
        start_idx = table_idx * sats_per_table
        end_idx = min(start_idx + sats_per_table, len(sorted_indices))
        
        if start_idx >= len(sorted_indices):
            break
        
        table = Table(show_header=True, header_style="bold magenta", box=None, padding=(0, 1))
        table.add_column("Satellite", style="cyan", width=25, no_wrap=True)
        table.add_column("Az", justify="right", style="yellow", width=7)
        table.add_column("El", justify="right", style="green", width=7)
        table.add_column("Rng", justify="right", style="blue", width=9)
        table.add_column("", width=6)
        
        for idx in sorted_indices[start_idx:end_idx]:
            name = str(tles[idx]).split('\n')[1].strip()
            if len(name) > 23:
                name = name[:22] + "…"
            
            el = elevations[idx]
            az = azimuths[idx]
            rng = ranges[idx]
            
            if el > 60:
                status = "[bold red]★[/bold red]"
            elif el > 30:
                status = "[yellow]▲[/yellow]"
            elif el > 10:
                status = "[cyan]●[/cyan]"
            else:
                status = "[dim]·[/dim]"
            
            table.add_row(
                name,
                f"{az:.1f}°",
                f"{el:.1f}°",
                f"{rng:.1f}km",
                status
            )
        
        tables.append(table)
    
    if tables:
        satellite_display = Columns(tables, equal=True, expand=True)
        
        total_shown = min(num_tables * sats_per_table, len(sorted_indices))
        if num_visible > total_shown:
            caption = Text(f"... and {num_visible - total_shown} more satellites", style="dim")
        else:
            caption = Text(f"Showing all {num_visible} visible satellites", style="dim")
        
        sat_panel = Panel(satellite_display, title=f"[bold]Visible Satellites[/bold]", 
                          subtitle=caption, border_style="cyan")
        layout["satellites"].update(sat_panel)
    else:
        layout["satellites"].update(Panel("No satellites visible", border_style="red"))
    
    footer = Panel(
        f"[yellow]Updating every {update_interval}s[/yellow] | [red]Press Ctrl+C to exit[/red]",
        style="dim"
    )
    layout["footer"].update(footer)
    
    return layout


def run_tracker(tle_file, observer, update_interval=0.01, require_gpu=True):
    """Run the real-time satellite tracker."""
    console = Console()
    
    # Check GPU availability
    if not torch.cuda.is_available():
        if require_gpu:
            console.print("[bold red]❌ ERROR: CUDA GPU not available![/bold red]")
            console.print("\nThis tracker requires an NVIDIA GPU with CUDA support.")
            console.print("Options:")
            console.print("  1. Install NVIDIA drivers and CUDA toolkit")
            console.print("  2. Run with --cpu flag to allow CPU fallback")
            console.print("\nCheck GPU with: nvidia-smi")
            return
        else:
            console.print("[yellow]⚠️  WARNING: Running on CPU (slow)[/yellow]")
            device = 'cpu'
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        console.print(f"[bold green]✓ GPU Detected:[/bold green] {gpu_name} ({gpu_mem:.1f} GB)")
        device = 'cuda'
    
    console.print(f"[yellow]Loading TLEs from:[/yellow] {tle_file}")
    
    tles = load(str(tle_file))
    
    console.print(f"[yellow]Initializing {len(tles)} satellites on {device.upper()}...[/yellow]")
    _, tle_batch = initialize_tle(tles, with_grad=False)
    
    console.print("[green]✓ Ready![/green]\n")
    time.sleep(1)
    
    iteration = 0
    
    with Live(console=console, refresh_per_second=100) as live:
        try:
            while True:
                iteration += 1
                
                target_time = datetime.now(timezone.utc)
                
                start_time = time.time()
                
                target_mjd = from_datetime_to_mjd(target_time)
                
                # Create tsinces tensor (dsgp4 handles GPU internally)
                tsinces = torch.tensor(
                    [(target_mjd - tle.date_mjd) * 1440.0 for tle in tles],
                    dtype=torch.float64
                )
                
                # Batch propagation - dsgp4 uses GPU acceleration internally
                states = propagate_batch(tle_batch, tsinces)
                
                prop_time = time.time() - start_time
                topo_start = time.time()
                
                positions_array = states[:, 0, :].cpu().numpy()
                
                jd = target_mjd + 2400000.5
                gst_result = gstime(torch.tensor([jd], dtype=torch.float64))
                gst = float(gst_result[0])
                
                observer_ecef = geodetic_to_ecef(observer.lat_rad, observer.lon_rad, observer.altitude)
                
                cos_gst = np.cos(gst)
                sin_gst = np.sin(gst)
                rotation = np.array([[cos_gst, sin_gst, 0], [-sin_gst, cos_gst, 0], [0, 0, 1]])
                
                positions_ecef = (rotation @ positions_array.T).T
                relative_ecef = positions_ecef - observer_ecef
                
                sin_lat = np.sin(observer.lat_rad)
                cos_lat = np.cos(observer.lat_rad)
                sin_lon = np.sin(observer.lon_rad)
                cos_lon = np.cos(observer.lon_rad)
                
                topo_rotation = np.array([
                    [sin_lat * cos_lon, sin_lat * sin_lon, -cos_lat],
                    [-sin_lon, cos_lon, 0],
                    [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat]
                ])
                
                sez = (topo_rotation @ relative_ecef.T).T
                
                ranges = np.linalg.norm(sez, axis=1)
                elevations = np.degrees(np.arcsin(sez[:, 2] / ranges))
                azimuths = np.degrees(np.arctan2(sez[:, 1], sez[:, 0]))
                azimuths[azimuths < 0] += 360
                
                visible_mask = elevations > 0
                visible_indices = np.where(visible_mask)[0]
                
                topo_time = time.time() - topo_start
                total_time = time.time() - start_time
                
                sorted_indices = visible_indices[np.argsort(-elevations[visible_indices])]
                
                num_visible = len(sorted_indices)
                num_high = np.sum(elevations[visible_indices] > 20)
                num_medium = np.sum((elevations[visible_indices] > 10) & (elevations[visible_indices] <= 20))
                num_low = np.sum(elevations[visible_indices] <= 10)
                
                display = make_display(
                    tles, tle_batch, observer, iteration, target_time,
                    num_visible, num_high, num_medium, num_low,
                    sorted_indices, elevations, azimuths, ranges,
                    prop_time, topo_time, total_time, update_interval
                )
                
                live.update(display)
                
                time.sleep(update_interval)
        
        except KeyboardInterrupt:
            pass
    
    console.print("\n[bold green]✓ Satellite Tracker Stopped[/bold green]\n")


def find_tle_file(tle_path=None):
    """Find the most recent TLE file."""
    if tle_path:
        tle_file = Path(tle_path)
        if tle_file.exists():
            return tle_file
    
    # Check for downloaded TLE (from make download-tle)
    if Path('starlink-tle-latest.tle').exists():
        return Path('starlink-tle-latest.tle')
    
    # Check for included TLE
    if Path('starlink-tle-2025-10-21-03-57-27.tle').exists():
        return Path('starlink-tle-2025-10-21-03-57-27.tle')
    
    # Find any TLE file in current directory
    tle_files = list(Path('.').glob('*.tle'))
    if tle_files:
        # Return most recently modified
        return max(tle_files, key=lambda p: p.stat().st_mtime)
    
    return None


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='GPU-accelerated real-time satellite tracker using dsgp4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s                              # Run with GPU (default location: San Francisco)
  %(prog)s --lat 40.7128 --lon -74.0060 # New York
  %(prog)s --cpu                         # Allow CPU fallback (slow)
  %(prog)s --interval 0.1                # Update every 0.1 seconds
        '''
    )
    parser.add_argument('--tle', type=str, default=None,
                        help='Path to TLE file (auto-detects .tle files if not specified)')
    parser.add_argument('--lat', type=float, default=37.7749,
                        help='Observer latitude in degrees (default: 37.7749 - San Francisco)')
    parser.add_argument('--lon', type=float, default=-122.4194,
                        help='Observer longitude in degrees (default: -122.4194 - San Francisco)')
    parser.add_argument('--alt', type=float, default=0.0,
                        help='Observer altitude in km (default: 0.0)')
    parser.add_argument('--interval', type=float, default=0.01,
                        help='Update interval in seconds (default: 0.01)')
    parser.add_argument('--cpu', action='store_true',
                        help='Allow CPU fallback if GPU not available (default: GPU required)')
    
    args = parser.parse_args()
    
    tle_file = find_tle_file(args.tle)
    
    if not tle_file:
        print(f"\n❌ Error: No TLE file found")
        print("\nOptions:")
        print("  1. Download TLE data: make download-tle")
        print("  2. Provide TLE file: poetry run python sattrack.py --tle your-file.txt")
        return 1
    
    print(f"📡 Using TLE file: {tle_file}")
    
    if not tle_file.exists():
        print(f"\n❌ Error: TLE file '{tle_file}' not found")
        return 1
    
    observer = Observer(latitude=args.lat, longitude=args.lon, altitude=args.alt)
    
    require_gpu = not args.cpu
    
    run_tracker(tle_file, observer, args.interval, require_gpu)
    
    return 0


if __name__ == "__main__":
    exit(main())
