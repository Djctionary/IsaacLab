#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""分析性能测试结果的脚本"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path


def analyze_performance_csv(csv_path: str, output_dir: str | None = None):
    """分析性能测试CSV文件并生成报告和可视化图表
    
    Args:
        csv_path: CSV文件路径
        output_dir: 输出目录，如果为None则使用CSV文件所在目录
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    # 检查是否有地形信息
    has_terrain_info = 'terrain_level' in df.columns and 'terrain_type' in df.columns
    
    # 创建报告文件
    report_path = os.path.join(output_dir, "performance_report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Performance Test Analysis Report\n")
        f.write("=" * 80 + "\n\n")
        
        # 基本信息
        f.write(f"Test File: {os.path.basename(csv_path)}\n")
        f.write(f"Total Iterations: {df['iteration'].max()}\n")
        f.write(f"Environments per Iteration: {len(df[df['iteration'] == 1])}\n")
        f.write(f"Total Test Environments: {len(df)}\n")
        
        if has_terrain_info:
            f.write(f"Terrain Levels: {df['terrain_level'].min()} - {df['terrain_level'].max()}\n")
            f.write(f"Terrain Types: {df['terrain_type'].min()} - {df['terrain_type'].max()}\n")
        f.write("\n")
        
        # Fall Statistics
        f.write("-" * 80 + "\n")
        f.write("Fall Statistics\n")
        f.write("-" * 80 + "\n")
        total_falls = df['fell_down'].sum()
        fall_rate = total_falls / len(df) * 100
        f.write(f"Total Falls: {total_falls}\n")
        f.write(f"Fall Rate: {fall_rate:.2f}%\n")
        f.write(f"Success Rate: {100 - fall_rate:.2f}%\n\n")
        
        # Fall statistics by iteration
        f.write("Fall Statistics by Iteration:\n")
        for iteration in sorted(df['iteration'].unique()):
            iter_df = df[df['iteration'] == iteration]
            iter_falls = iter_df['fell_down'].sum()
            iter_rate = iter_falls / len(iter_df) * 100
            f.write(f"  Iteration {iteration}: {iter_falls}/{len(iter_df)} ({iter_rate:.2f}%)\n")
        f.write("\n")
        
        # Terrain-based statistics
        if has_terrain_info:
            f.write("-" * 80 + "\n")
            f.write("Terrain-based Performance Analysis\n")
            f.write("-" * 80 + "\n")
            
            # Fall rate by terrain level
            f.write("Fall Rate by Terrain Level:\n")
            for level in sorted(df['terrain_level'].unique()):
                level_df = df[df['terrain_level'] == level]
                level_falls = level_df['fell_down'].sum()
                level_rate = level_falls / len(level_df) * 100
                f.write(f"  Level {level}: {level_falls}/{len(level_df)} ({level_rate:.2f}%)\n")
            f.write("\n")
            
            # Fall rate by terrain type
            f.write("Fall Rate by Terrain Type:\n")
            for terrain_type in sorted(df['terrain_type'].unique()):
                type_df = df[df['terrain_type'] == terrain_type]
                type_falls = type_df['fell_down'].sum()
                type_rate = type_falls / len(type_df) * 100
                f.write(f"  Type {terrain_type}: {type_falls}/{len(type_df)} ({type_rate:.2f}%)\n")
            f.write("\n")
            
            # Performance by terrain level and type
            f.write("Performance by Terrain Level and Type:\n")
            for level in sorted(df['terrain_level'].unique()):
                for terrain_type in sorted(df['terrain_type'].unique()):
                    subset_df = df[(df['terrain_level'] == level) & (df['terrain_type'] == terrain_type)]
                    if len(subset_df) > 0:
                        subset_falls = subset_df['fell_down'].sum()
                        subset_rate = subset_falls / len(subset_df) * 100
                        avg_lin_error = subset_df['lin_vel_tracking_error'].mean()
                        avg_ang_error = subset_df['ang_vel_tracking_error'].mean()
                        f.write(f"  Level {level}, Type {terrain_type}: {subset_falls}/{len(subset_df)} ({subset_rate:.2f}%) | "
                               f"Lin Error: {avg_lin_error:.4f} | Ang Error: {avg_ang_error:.4f}\n")
            f.write("\n")
        
        # Velocity tracking performance
        f.write("-" * 80 + "\n")
        f.write("Velocity Tracking Performance (All Environments)\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Linear Velocity Tracking Error: {df['lin_vel_tracking_error'].mean():.4f} ± {df['lin_vel_tracking_error'].std():.4f} m/s\n")
        f.write(f"Average Angular Velocity Tracking Error: {df['ang_vel_tracking_error'].mean():.4f} ± {df['ang_vel_tracking_error'].std():.4f} rad/s\n\n")
        
        f.write(f"Linear Velocity Tracking Error Median: {df['lin_vel_tracking_error'].median():.4f} m/s\n")
        f.write(f"Angular Velocity Tracking Error Median: {df['ang_vel_tracking_error'].median():.4f} rad/s\n\n")
        
        f.write(f"Linear Velocity Tracking Error Max: {df['lin_vel_tracking_error'].max():.4f} m/s\n")
        f.write(f"Linear Velocity Tracking Error Min: {df['lin_vel_tracking_error'].min():.4f} m/s\n\n")
        
        # Analyze only successful environments
        success_df = df[df['fell_down'] == 0]
        if len(success_df) > 0:
            f.write("-" * 80 + "\n")
            f.write("Velocity Tracking Performance (Successful Environments Only)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average Linear Velocity Tracking Error: {success_df['lin_vel_tracking_error'].mean():.4f} ± {success_df['lin_vel_tracking_error'].std():.4f} m/s\n")
            f.write(f"Average Angular Velocity Tracking Error: {success_df['ang_vel_tracking_error'].mean():.4f} ± {success_df['ang_vel_tracking_error'].std():.4f} rad/s\n\n")
            
            f.write(f"Actual Average Linear Velocity Norm: {success_df['avg_actual_lin_vel_xy_norm'].mean():.4f} ± {success_df['avg_actual_lin_vel_xy_norm'].std():.4f} m/s\n")
            f.write(f"Command Average Linear Velocity Norm: {success_df['avg_command_lin_vel_xy_norm'].mean():.4f} ± {success_df['avg_command_lin_vel_xy_norm'].std():.4f} m/s\n\n")
            
            f.write(f"Actual Average Angular Velocity: {success_df['avg_actual_ang_vel_z'].mean():.4f} ± {success_df['avg_actual_ang_vel_z'].std():.4f} rad/s\n")
            f.write(f"Command Average Angular Velocity: {success_df['avg_command_ang_vel_z'].mean():.4f} ± {success_df['avg_command_ang_vel_z'].std():.4f} rad/s\n\n")
        
        # Episode length statistics
        f.write("-" * 80 + "\n")
        f.write("Episode Length Statistics\n")
        f.write("-" * 80 + "\n")
        f.write(f"Average Steps: {df['episode_length'].mean():.2f} ± {df['episode_length'].std():.2f}\n")
        f.write(f"Median Steps: {df['episode_length'].median():.2f}\n")
        f.write(f"Max Steps: {df['episode_length'].max()}\n")
        f.write(f"Min Steps: {df['episode_length'].min()}\n\n")
        
    print(f"Analysis report saved to: {report_path}")
    
    # Generate visualization charts
    create_visualizations(df, output_dir)


def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create visualization charts
    
    Args:
        df: Performance data DataFrame
        output_dir: Output directory
    """
    import matplotlib
    matplotlib.use('Agg')  # Use non-GUI backend
    
    # Set font for better display
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Check if terrain information is available
    has_terrain_info = 'terrain_level' in df.columns and 'terrain_type' in df.columns
    
    # 1. Fall rate by iteration
    fig, ax = plt.subplots(figsize=(10, 6))
    iterations = sorted(df['iteration'].unique())
    fall_rates = []
    for iteration in iterations:
        iter_df = df[df['iteration'] == iteration]
        fall_rate = iter_df['fell_down'].sum() / len(iter_df) * 100
        fall_rates.append(fall_rate)
    
    ax.bar(iterations, fall_rates, color='coral', alpha=0.7)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Fall Rate (%)', fontsize=12)
    ax.set_title('Fall Rate by Iteration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fall_rate_by_iteration.png'), dpi=300)
    plt.close()
    print(f"Chart saved: fall_rate_by_iteration.png")
    
    # 2. 速度跟踪误差分布
    success_df = df[df['fell_down'] == 0]
    if len(success_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Linear velocity tracking error
        axes[0].hist(success_df['lin_vel_tracking_error'], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0].axvline(success_df['lin_vel_tracking_error'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f"Mean: {success_df['lin_vel_tracking_error'].mean():.4f}")
        axes[0].set_xlabel('Linear Velocity Tracking Error (m/s)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Linear Velocity Tracking Error Distribution (Successful Only)', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Angular velocity tracking error
        axes[1].hist(success_df['ang_vel_tracking_error'], bins=50, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[1].axvline(success_df['ang_vel_tracking_error'].mean(), color='red', linestyle='--', 
                       linewidth=2, label=f"Mean: {success_df['ang_vel_tracking_error'].mean():.4f}")
        axes[1].set_xlabel('Angular Velocity Tracking Error (rad/s)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Angular Velocity Tracking Error Distribution (Successful Only)', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tracking_error_distribution.png'), dpi=300)
        plt.close()
        print(f"Chart saved: tracking_error_distribution.png")
    
    # 3. 实际速度 vs 期望速度散点图
    if len(success_df) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Linear velocity comparison
        axes[0].scatter(success_df['avg_command_lin_vel_xy_norm'], 
                       success_df['avg_actual_lin_vel_xy_norm'], 
                       alpha=0.5, s=10, color='blue')
        max_vel = max(success_df['avg_command_lin_vel_xy_norm'].max(), 
                     success_df['avg_actual_lin_vel_xy_norm'].max())
        min_vel = min(success_df['avg_command_lin_vel_xy_norm'].min(), 
                     success_df['avg_actual_lin_vel_xy_norm'].min())
        axes[0].plot([min_vel, max_vel], [min_vel, max_vel], 'r--', linewidth=2, label='Perfect Tracking')
        axes[0].set_xlabel('Command Linear Velocity Norm (m/s)', fontsize=12)
        axes[0].set_ylabel('Actual Linear Velocity Norm (m/s)', fontsize=12)
        axes[0].set_title('Linear Velocity Tracking Performance', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal', adjustable='box')
        
        # Angular velocity comparison
        axes[1].scatter(success_df['avg_command_ang_vel_z'], 
                       success_df['avg_actual_ang_vel_z'], 
                       alpha=0.5, s=10, color='green')
        max_ang = max(success_df['avg_command_ang_vel_z'].max(), 
                     success_df['avg_actual_ang_vel_z'].max())
        min_ang = min(success_df['avg_command_ang_vel_z'].min(), 
                     success_df['avg_actual_ang_vel_z'].min())
        axes[1].plot([min_ang, max_ang], [min_ang, max_ang], 'r--', linewidth=2, label='Perfect Tracking')
        axes[1].set_xlabel('Command Angular Velocity (rad/s)', fontsize=12)
        axes[1].set_ylabel('Actual Angular Velocity (rad/s)', fontsize=12)
        axes[1].set_title('Angular Velocity Tracking Performance', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'velocity_tracking_comparison.png'), dpi=300)
        plt.close()
        print(f"Chart saved: velocity_tracking_comparison.png")
    
    # 4. Tracking error box plots (by iteration)
    if len(success_df) > 0 and success_df['iteration'].nunique() > 1:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data
        lin_vel_errors = [success_df[success_df['iteration'] == i]['lin_vel_tracking_error'].values 
                         for i in sorted(success_df['iteration'].unique())]
        ang_vel_errors = [success_df[success_df['iteration'] == i]['ang_vel_tracking_error'].values 
                         for i in sorted(success_df['iteration'].unique())]
        
        # Linear velocity tracking error box plot
        bp1 = axes[0].boxplot(lin_vel_errors, labels=sorted(success_df['iteration'].unique()),
                             patch_artist=True, showmeans=True)
        for patch in bp1['boxes']:
            patch.set_facecolor('lightblue')
        axes[0].set_xlabel('Iteration', fontsize=12)
        axes[0].set_ylabel('Linear Velocity Tracking Error (m/s)', fontsize=12)
        axes[0].set_title('Linear Velocity Tracking Error Distribution (by Iteration)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Angular velocity tracking error box plot
        bp2 = axes[1].boxplot(ang_vel_errors, labels=sorted(success_df['iteration'].unique()),
                             patch_artist=True, showmeans=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightgreen')
        axes[1].set_xlabel('Iteration', fontsize=12)
        axes[1].set_ylabel('Angular Velocity Tracking Error (rad/s)', fontsize=12)
        axes[1].set_title('Angular Velocity Tracking Error Distribution (by Iteration)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'tracking_error_by_iteration.png'), dpi=300)
        plt.close()
        print(f"Chart saved: tracking_error_by_iteration.png")
    
    # 5. Terrain-based visualizations (if terrain info is available)
    if has_terrain_info:
        # Fall rate by terrain level
        fig, ax = plt.subplots(figsize=(10, 6))
        levels = sorted(df['terrain_level'].unique())
        level_fall_rates = []
        for level in levels:
            level_df = df[df['terrain_level'] == level]
            fall_rate = level_df['fell_down'].sum() / len(level_df) * 100
            level_fall_rates.append(fall_rate)
        
        ax.bar(levels, level_fall_rates, color='steelblue', alpha=0.7)
        ax.set_xlabel('Terrain Level', fontsize=12)
        ax.set_ylabel('Fall Rate (%)', fontsize=12)
        ax.set_title('Fall Rate by Terrain Level', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fall_rate_by_terrain_level.png'), dpi=300)
        plt.close()
        print(f"Chart saved: fall_rate_by_terrain_level.png")
        
        # Fall rate by terrain type
        fig, ax = plt.subplots(figsize=(10, 6))
        types = sorted(df['terrain_type'].unique())
        type_fall_rates = []
        for terrain_type in types:
            type_df = df[df['terrain_type'] == terrain_type]
            fall_rate = type_df['fell_down'].sum() / len(type_df) * 100
            type_fall_rates.append(fall_rate)
        
        ax.bar(types, type_fall_rates, color='forestgreen', alpha=0.7)
        ax.set_xlabel('Terrain Type', fontsize=12)
        ax.set_ylabel('Fall Rate (%)', fontsize=12)
        ax.set_title('Fall Rate by Terrain Type', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fall_rate_by_terrain_type.png'), dpi=300)
        plt.close()
        print(f"Chart saved: fall_rate_by_terrain_type.png")
        
        # Performance heatmap: terrain level vs type
        if len(success_df) > 0:
            # Create pivot table for fall rate
            pivot_fall = df.pivot_table(values='fell_down', index='terrain_level', 
                                      columns='terrain_type', aggfunc='mean') * 100
            
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(pivot_fall.values, cmap='Reds', aspect='auto')
            ax.set_xticks(range(len(pivot_fall.columns)))
            ax.set_yticks(range(len(pivot_fall.index)))
            ax.set_xticklabels(pivot_fall.columns)
            ax.set_yticklabels(pivot_fall.index)
            ax.set_xlabel('Terrain Type', fontsize=12)
            ax.set_ylabel('Terrain Level', fontsize=12)
            ax.set_title('Fall Rate Heatmap: Terrain Level vs Type', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Fall Rate (%)', fontsize=12)
            
            # Add text annotations
            for i in range(len(pivot_fall.index)):
                for j in range(len(pivot_fall.columns)):
                    text = ax.text(j, i, f'{pivot_fall.iloc[i, j]:.1f}%',
                                 ha="center", va="center", color="black", fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'fall_rate_heatmap.png'), dpi=300)
            plt.close()
            print(f"Chart saved: fall_rate_heatmap.png")
            
            # Performance heatmap: tracking error by terrain
            pivot_lin_error = success_df.pivot_table(values='lin_vel_tracking_error', 
                                                    index='terrain_level', 
                                                    columns='terrain_type', aggfunc='mean')
            
            fig, ax = plt.subplots(figsize=(12, 8))
            im = ax.imshow(pivot_lin_error.values, cmap='Blues', aspect='auto')
            ax.set_xticks(range(len(pivot_lin_error.columns)))
            ax.set_yticks(range(len(pivot_lin_error.index)))
            ax.set_xticklabels(pivot_lin_error.columns)
            ax.set_yticklabels(pivot_lin_error.index)
            ax.set_xlabel('Terrain Type', fontsize=12)
            ax.set_ylabel('Terrain Level', fontsize=12)
            ax.set_title('Linear Velocity Tracking Error Heatmap', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Tracking Error (m/s)', fontsize=12)
            
            # Add text annotations
            for i in range(len(pivot_lin_error.index)):
                for j in range(len(pivot_lin_error.columns)):
                    if not np.isnan(pivot_lin_error.iloc[i, j]):
                        text = ax.text(j, i, f'{pivot_lin_error.iloc[i, j]:.3f}',
                                     ha="center", va="center", color="white", fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'tracking_error_heatmap.png'), dpi=300)
            plt.close()
            print(f"Chart saved: tracking_error_heatmap.png")
            
            # Box plot: tracking error by terrain level
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # Linear velocity tracking error by terrain level
            terrain_levels = sorted(success_df['terrain_level'].unique())
            lin_error_data = [success_df[success_df['terrain_level'] == level]['lin_vel_tracking_error'].values 
                            for level in terrain_levels]
            
            bp1 = axes[0].boxplot(lin_error_data, labels=terrain_levels, patch_artist=True, showmeans=True)
            for patch in bp1['boxes']:
                patch.set_facecolor('lightblue')
            axes[0].set_xlabel('Terrain Level', fontsize=12)
            axes[0].set_ylabel('Linear Velocity Tracking Error (m/s)', fontsize=12)
            axes[0].set_title('Linear Velocity Tracking Error by Terrain Level', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3)
            
            # Angular velocity tracking error by terrain level
            ang_error_data = [success_df[success_df['terrain_level'] == level]['ang_vel_tracking_error'].values 
                            for level in terrain_levels]
            
            bp2 = axes[1].boxplot(ang_error_data, labels=terrain_levels, patch_artist=True, showmeans=True)
            for patch in bp2['boxes']:
                patch.set_facecolor('lightgreen')
            axes[1].set_xlabel('Terrain Level', fontsize=12)
            axes[1].set_ylabel('Angular Velocity Tracking Error (rad/s)', fontsize=12)
            axes[1].set_title('Angular Velocity Tracking Error by Terrain Level', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'tracking_error_by_terrain_level.png'), dpi=300)
            plt.close()
            print(f"Chart saved: tracking_error_by_terrain_level.png")
    
    print(f"\nAll visualization charts saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Analyze performance test results")
    parser.add_argument("csv_path", type=str, help="Performance test CSV file path")
    parser.add_argument("--output_dir", type=str, default=None, 
                       help="Output directory (default: same as CSV file directory)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_path):
        print(f"Error: File not found {args.csv_path}")
        return
    
    analyze_performance_csv(args.csv_path, args.output_dir)
    print("\nAnalysis completed!")


if __name__ == "__main__":
    main()

