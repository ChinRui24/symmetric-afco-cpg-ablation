"""
对称AFCO CPG 消融实验框架
Ablation Study Framework for Symmetric AFCO CPG

用于SCI论文的理论验证
Theory Validation for SCI Publication

消融实验设计：
1. Full Model - 完整模型（基线）
2. w/o Symmetric PRC - 移除对称相位修正
3. w/o GRF Weighting - 移除接触力加权
4. w/o Adaptive Coupling - 移除自适应耦合
5. w/o Frequency Adaptation - 移除频率自适应
6. w/o Shock Suppression - 移除冲击抑制
7. Minimal Model - 最小化模型（只保留基础CPG）
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非GUI后端
import seaborn as sns
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import pandas as pd
from pathlib import Path
import json

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置科研论文风格
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def wrap_to_pi(x):
    """Wrap angle to [-pi, pi]"""
    return (x + np.pi) % (2 * np.pi) - np.pi


def lpf(x, prev, alpha):
    """Low-pass filter"""
    return (1 - alpha) * prev + alpha * x


@dataclass
class AblationConfig:
    """消融实验配置"""
    name: str
    enable_symmetric_prc: bool = True      # 对称PRC修正
    enable_grf_weighting: bool = True      # GRF加权
    enable_adaptive_coupling: bool = True  # 自适应耦合
    enable_frequency_adapt: bool = True    # 频率自适应
    enable_shock_suppression: bool = True  # 冲击抑制
    coordination_mode: str = 'diagonal'    # 协调模式
    
    def get_description(self) -> str:
        """获取配置描述"""
        disabled = []
        if not self.enable_symmetric_prc:
            disabled.append("w/o Sym-PRC")
        if not self.enable_grf_weighting:
            disabled.append("w/o GRF-Weight")
        if not self.enable_adaptive_coupling:
            disabled.append("w/o Adapt-Coupling")
        if not self.enable_frequency_adapt:
            disabled.append("w/o Freq-Adapt")
        if not self.enable_shock_suppression:
            disabled.append("w/o Shock-Suppress")
        
        if not disabled:
            return "Full Model"
        return ", ".join(disabled)


@dataclass
class SimulationMetrics:
    """仿真性能指标"""
    # 相位同步性
    phase_synchronization: float  # 相位同步误差（越小越好）
    phase_convergence_time: float  # 相位收敛时间（越小越好）
    phase_stability: float  # 相位稳定性（越大越好）
    
    # 步态规律性
    gait_regularity: float  # 步态规律性（越大越好）
    stride_consistency: float  # 步幅一致性（越大越好）
    
    # 姿态稳定性
    body_roll_std: float  # 侧倾标准差（越小越好）
    body_pitch_std: float  # 俯仰标准差（越小越好）
    body_oscillation: float  # 身体振荡幅度（越小越好）
    
    # 鲁棒性
    disturbance_recovery_time: float  # 扰动恢复时间（越小越好）
    disturbance_deviation: float  # 扰动偏差（越小越好）
    
    # 能量效率
    frequency_variation: float  # 频率变化（适度）
    coupling_efficiency: float  # 耦合效率（越大越好）
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SimplifiedSymmetricCPG:
    """
    简化的对称CPG（纯Python实现，用于消融实验）
    
    不依赖PyBullet，只模拟CPG内部动力学
    """
    
    def __init__(self, config: AblationConfig, ftype: int = 2):
        """
        初始化
        
        Args:
            config: 消融配置
            ftype: 步态类型 (1=walk, 2=trot, 3=pace, 4=bound, 5=pronk)
        """
        self.config = config
        self.ftype = ftype
        
        # 基础参数（继承原始AFCO）
        self.omega_base = 2 * np.pi / 2  # 基础频率
        self.omega_min = 0.5
        self.omega_max = 10.0
        
        self.coupling_base = 0.55
        self.coupling_min = 0.0
        self.coupling_max = 3.0
        self.beta = 2.0  # 耦合增益
        self.c_rate = 8.0  # 耦合变化率
        
        self.alpha_omega = 4.0  # 频率适应率
        self.gamma_shock = 1.0  # 冲击增益
        
        # 姿态参数
        self.a_theta = 3.0
        self.b_theta = 1.25
        self.a_psi = 3.0
        self.b_psi = 1.25
        self.g_theta = 0.15
        self.g_psi = 0.15
        
        # 接触力参数
        self.grf_bias = 0.25
        self.grf_gain = 0.75
        
        # PRC参数
        self.eps_prc = 0.8
        self.grf_weight_prc = 0.3 if config.enable_grf_weighting else 0.0
        
        # 滤波参数
        self.grf_lpf_alpha = 0.03
        self.err_lpf_alpha = 0.05
        self.shock_lpf_alpha = 0.01
        self.k_contact = 6.0
        
        self.mult_min = 0.25
        self.mult_max = 2.0
        
        # 初始化状态
        self.phi = self.target_phases(ftype)  # [FR, FL, RL, RR]
        self.omega = np.ones(4) * self.omega_base
        self.coupling = np.ones(4) * self.coupling_base
        self.err_f = np.zeros(4)
        self.grf_f = np.zeros(4)
        self.grf_s_lpf = np.zeros(4)
        self.gate_s_lpf = np.zeros(4)
        self.theta = 0.0
        self.psi = 0.0
        
        # 相位差矩阵
        self.set_phase_matrix(ftype)
        
        # 历史记录（用于计算指标）
        self.history = {
            'phi': [],
            'omega': [],
            'coupling': [],
            'theta': [],
            'psi': [],
            'grf': [],
            'phase_error': [],
        }
    
    def target_phases(self, ftype: int) -> np.ndarray:
        """目标相位模式"""
        if ftype == 1:  # walk
            return np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi])
        elif ftype == 2:  # trot
            return np.array([0.0, np.pi, 0.0, np.pi])
        elif ftype == 3:  # pace
            return np.array([0.0, np.pi, np.pi, 0.0])
        elif ftype == 4:  # bound
            return np.array([np.pi, np.pi, 0.0, 0.0])
        elif ftype == 5:  # pronk
            return np.zeros(4)
        else:
            return np.array([0.0, np.pi, 0.0, np.pi])
    
    def set_phase_matrix(self, ftype: int):
        """设置相位差矩阵"""
        phi_target = self.target_phases(ftype)
        self.dphi_matrix = np.zeros((4, 4))
        for i in range(4):
            for j in range(4):
                self.dphi_matrix[i, j] = wrap_to_pi(phi_target[j] - phi_target[i])
    
    def smooth_contact_gate(self, phi: np.ndarray) -> np.ndarray:
        """平滑接触门信号"""
        return 0.5 * (1.0 + np.tanh(self.k_contact * np.sin(phi)))
    
    def load_weights_from_attitude(self, theta: float, psi: float) -> np.ndarray:
        """从姿态计算负载权重"""
        wF = 0.5 * (1.0 + self.g_theta * theta)
        wR = 0.5 * (1.0 - self.g_theta * theta)
        wRight = 0.5 * (1.0 + self.g_psi * psi)
        wLeft = 0.5 * (1.0 - self.g_psi * psi)
        
        wF = np.clip(wF, 0.2, 0.8)
        wR = np.clip(wR, 0.2, 0.8)
        wRight = np.clip(wRight, 0.2, 0.8)
        wLeft = np.clip(wLeft, 0.2, 0.8)
        
        sF = wF / (wF + wR)
        sR = wR / (wF + wR)
        sRight = wRight / (wRight + wLeft)
        sLeft = wLeft / (wRight + wLeft)
        
        w = np.array([
            sF * sRight,   # FR
            sF * sLeft,    # FL
            sR * sLeft,    # RL
            sR * sRight,   # RR
        ])
        
        w /= np.sum(w)
        return w
    
    def get_coordination_partners(self, leg_idx: int) -> tuple:
        """获取协调伙伴"""
        if self.config.coordination_mode == 'diagonal':
            diagonal_map = {0: 2, 1: 3, 2: 0, 3: 1}
            partner = diagonal_map[leg_idx]
            return np.array([partner]), np.array([1.0])
        elif self.config.coordination_mode == 'all':
            partners = np.array([i for i in range(4) if i != leg_idx])
            weights = self.grf_f[partners]
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(partners)) / len(partners)
            return partners, weights
        else:
            raise ValueError(f"Unknown mode: {self.config.coordination_mode}")
    
    def dynamics(self, dt: float = 0.002):
        """
        CPG动力学（单步）
        
        根据消融配置选择性启用/禁用功能
        """
        phi = self.phi
        
        # 1. 接触门
        gate = self.smooth_contact_gate(phi)
        
        # 2. 姿态动力学（简化模拟）
        front = gate[0] + gate[1]
        rear = gate[2] + gate[3]
        right = gate[0] + gate[3]
        left = gate[1] + gate[2]
        
        theta_dot = -self.a_theta * self.theta + self.b_theta * (front - rear)
        psi_dot = -self.a_psi * self.psi + self.b_psi * (right - left)
        
        self.theta += theta_dot * dt
        self.psi += psi_dot * dt
        self.theta = np.clip(self.theta, -0.3, 0.3)
        self.psi = np.clip(self.psi, -0.3, 0.3)
        
        # 3. 负载权重
        w = self.load_weights_from_attitude(self.theta, self.psi)
        grf_raw = np.clip(gate * (self.grf_bias + self.grf_gain * w), 0.0, 1.0)
        self.grf_f = lpf(grf_raw, self.grf_f, self.grf_lpf_alpha)
        
        # 4. 相位误差
        err = np.zeros(4)
        for i in range(4):
            partners, weights = self.get_coordination_partners(i)
            phase_errors = []
            
            for j, w_j in zip(partners, weights):
                target_dphi = self.dphi_matrix[i, j]
                actual_dphi = wrap_to_pi(phi[j] - phi[i])
                phase_err = wrap_to_pi(actual_dphi - target_dphi)
                phase_errors.append(w_j * 0.5 * (1.0 - np.cos(phase_err)))
            
            err[i] = sum(phase_errors) if phase_errors else 0.0
        
        self.err_f = lpf(err, self.err_f, self.err_lpf_alpha)
        
        # 5. 自适应耦合（可消融）
        if self.config.enable_adaptive_coupling:
            c_tar = self.coupling_base * (1.0 + self.beta * self.err_f)
            dc = np.clip(c_tar - self.coupling, -self.c_rate * dt, self.c_rate * dt)
            self.coupling = self.coupling + gate * dc
            self.coupling = np.clip(self.coupling, self.coupling_min, self.coupling_max)
        else:
            self.coupling = np.ones(4) * self.coupling_base
        
        # 6. 冲击抑制（可消融）
        if self.config.enable_shock_suppression:
            self.grf_s_lpf = lpf(grf_raw, self.grf_s_lpf, self.shock_lpf_alpha)
            self.gate_s_lpf = lpf(gate, self.gate_s_lpf, self.shock_lpf_alpha)
            shock = np.clip((grf_raw - self.grf_s_lpf), 0.0, 1.0) * \
                    np.clip((gate - self.gate_s_lpf), 0.0, 1.0)
        else:
            shock = np.zeros(4)
        
        # 7. 频率自适应（可消融）
        if self.config.enable_frequency_adapt:
            omega_cmd = self.omega_base
            domega = self.alpha_omega * (omega_cmd - self.omega) - self.gamma_shock * shock
            self.omega = self.omega + domega * dt
            self.omega = np.clip(self.omega, self.omega_min, self.omega_max)
        else:
            self.omega = np.ones(4) * self.omega_base
        
        # 8. 基础相位动力学
        s = 0.5 * (1.0 + np.cos(phi))
        mult = np.clip(1.0 - self.coupling * self.grf_f * s, self.mult_min, self.mult_max)
        phi_dot = 2 * np.pi * (self.omega * mult)
        
        # 9. 对称PRC修正（可消融 - 关键创新）
        if self.config.enable_symmetric_prc:
            prc_correction = np.zeros(4)
            
            for i in range(4):
                partners, weights = self.get_coordination_partners(i)
                
                for j, w_j in zip(partners, weights):
                    target_dphi = self.dphi_matrix[i, j]
                    actual_dphi = wrap_to_pi(phi[j] - phi[i])
                    phase_err = wrap_to_pi(actual_dphi - target_dphi)
                    
                    grf_trust = self.grf_f[j] if self.grf_weight_prc > 0 else 1.0
                    correction = w_j * grf_trust * np.sin(phase_err)
                    prc_correction[i] += correction
                
                prc_correction[i] *= gate[i]
            
            phi_dot += 2 * np.pi * self.eps_prc * prc_correction
        
        # 更新相位
        self.phi = (self.phi + phi_dot * dt) % (2 * np.pi)
        
        # 记录历史
        self.history['phi'].append(self.phi.copy())
        self.history['omega'].append(self.omega.copy())
        self.history['coupling'].append(self.coupling.copy())
        self.history['theta'].append(self.theta)
        self.history['psi'].append(self.psi)
        self.history['grf'].append(self.grf_f.copy())
        self.history['phase_error'].append(self.err_f.copy())
    
    def reset(self):
        """重置状态"""
        self.phi = self.target_phases(self.ftype)
        self.omega = np.ones(4) * self.omega_base
        self.coupling = np.ones(4) * self.coupling_base
        self.err_f = np.zeros(4)
        self.grf_f = np.zeros(4)
        self.grf_s_lpf = np.zeros(4)
        self.gate_s_lpf = np.zeros(4)
        self.theta = 0.0
        self.psi = 0.0
        self.history = {k: [] for k in self.history.keys()}
    
    def warmup(self, duration: float = 2.0, dt: float = 0.002):
        """预热"""
        num_steps = int(duration / dt)
        for _ in range(num_steps):
            self.dynamics(dt)
    
    def apply_phase_disturbance(self, magnitude: float = 0.5):
        """施加相位扰动（用于鲁棒性测试）"""
        disturbance = np.random.uniform(-magnitude, magnitude, 4)
        self.phi = (self.phi + disturbance) % (2 * np.pi)
    
    def apply_attitude_disturbance(self, theta_dist: float = 0.2, psi_dist: float = 0.2):
        """施加姿态扰动"""
        self.theta += theta_dist
        self.psi += psi_dist


class AblationStudyRunner:
    """消融实验执行器"""
    
    def __init__(self, output_dir: str = "/home/claude/ablation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义所有消融配置
        self.configs = self._create_ablation_configs()
        
        # 结果存储
        self.results: Dict[str, SimulationMetrics] = {}
    
    def _create_ablation_configs(self) -> List[AblationConfig]:
        """创建所有消融实验配置"""
        return [
            # 1. 完整模型（基线）
            AblationConfig(
                name="Full Model",
                enable_symmetric_prc=True,
                enable_grf_weighting=True,
                enable_adaptive_coupling=True,
                enable_frequency_adapt=True,
                enable_shock_suppression=True,
            ),
            
            # 2. 移除对称PRC（最关键的创新）
            AblationConfig(
                name="w/o Symmetric PRC",
                enable_symmetric_prc=False,
                enable_grf_weighting=True,
                enable_adaptive_coupling=True,
                enable_frequency_adapt=True,
                enable_shock_suppression=True,
            ),
            
            # 3. 移除GRF加权
            AblationConfig(
                name="w/o GRF Weighting",
                enable_symmetric_prc=True,
                enable_grf_weighting=False,
                enable_adaptive_coupling=True,
                enable_frequency_adapt=True,
                enable_shock_suppression=True,
            ),
            
            # 4. 移除自适应耦合
            AblationConfig(
                name="w/o Adaptive Coupling",
                enable_symmetric_prc=True,
                enable_grf_weighting=True,
                enable_adaptive_coupling=False,
                enable_frequency_adapt=True,
                enable_shock_suppression=True,
            ),
            
            # 5. 移除频率自适应
            AblationConfig(
                name="w/o Frequency Adapt",
                enable_symmetric_prc=True,
                enable_grf_weighting=True,
                enable_adaptive_coupling=True,
                enable_frequency_adapt=False,
                enable_shock_suppression=True,
            ),
            
            # 6. 移除冲击抑制
            AblationConfig(
                name="w/o Shock Suppress",
                enable_symmetric_prc=True,
                enable_grf_weighting=True,
                enable_adaptive_coupling=True,
                enable_frequency_adapt=True,
                enable_shock_suppression=False,
            ),
            
            # 7. 最小化模型（只保留基础CPG）
            AblationConfig(
                name="Minimal Model",
                enable_symmetric_prc=False,
                enable_grf_weighting=False,
                enable_adaptive_coupling=False,
                enable_frequency_adapt=False,
                enable_shock_suppression=False,
            ),
        ]
    
    def run_single_trial(
        self,
        config: AblationConfig,
        duration: float = 20.0,
        dt: float = 0.002,
        apply_disturbance: bool = True,
        disturbance_time: float = 10.0
    ) -> SimulationMetrics:
        """
        运行单次试验
        
        Args:
            config: 消融配置
            duration: 仿真时长（秒）
            dt: 时间步长
            apply_disturbance: 是否施加扰动
            disturbance_time: 扰动时刻
        
        Returns:
            性能指标
        """
        print(f"\n运行试验: {config.name}")
        print(f"  配置: {config.get_description()}")
        
        # 创建CPG
        cpg = SimplifiedSymmetricCPG(config, ftype=2)  # Trot步态
        
        # 预热
        print("  预热中...")
        cpg.warmup(duration=2.0, dt=dt)
        cpg.history = {k: [] for k in cpg.history.keys()}  # 清空预热数据
        
        # 主仿真
        print("  运行仿真...")
        num_steps = int(duration / dt)
        disturbance_step = int(disturbance_time / dt)
        
        for step in range(num_steps):
            # 施加扰动
            if apply_disturbance and step == disturbance_step:
                print(f"    施加扰动 @ t={disturbance_time}s")
                cpg.apply_phase_disturbance(magnitude=0.8)
                cpg.apply_attitude_disturbance(theta_dist=0.15, psi_dist=0.15)
            
            # 动力学
            cpg.dynamics(dt)
        
        # 计算指标
        print("  计算指标...")
        metrics = self._compute_metrics(cpg, dt, disturbance_time, duration)
        
        return metrics
    
    def _compute_metrics(
        self,
        cpg: SimplifiedSymmetricCPG,
        dt: float,
        disturbance_time: float,
        total_duration: float
    ) -> SimulationMetrics:
        """计算性能指标"""
        
        # 提取历史数据
        phi_history = np.array(cpg.history['phi'])  # [T, 4]
        omega_history = np.array(cpg.history['omega'])
        coupling_history = np.array(cpg.history['coupling'])
        theta_history = np.array(cpg.history['theta'])
        psi_history = np.array(cpg.history['psi'])
        phase_err_history = np.array(cpg.history['phase_error'])
        
        T = len(phi_history)
        time = np.arange(T) * dt
        
        # 目标相位
        phi_target = cpg.target_phases(cpg.ftype)
        
        # 1. 相位同步性（整个过程）
        phase_sync_errors = []
        for t in range(T):
            phi_t = phi_history[t]
            for i in range(4):
                for j in range(i+1, 4):
                    target_dphi = wrap_to_pi(phi_target[j] - phi_target[i])
                    actual_dphi = wrap_to_pi(phi_t[j] - phi_t[i])
                    err = abs(wrap_to_pi(actual_dphi - target_dphi))
                    phase_sync_errors.append(err)
        
        phase_synchronization = np.mean(phase_sync_errors)
        
        # 2. 相位收敛时间（前5秒）
        convergence_threshold = 0.1  # rad
        convergence_window = int(5.0 / dt)
        convergence_window = min(convergence_window, T)
        
        convergence_time = total_duration  # 默认未收敛
        for t in range(convergence_window):
            phi_t = phi_history[t]
            max_err = 0
            for i in range(4):
                for j in range(i+1, 4):
                    target_dphi = wrap_to_pi(phi_target[j] - phi_target[i])
                    actual_dphi = wrap_to_pi(phi_t[j] - phi_t[i])
                    err = abs(wrap_to_pi(actual_dphi - target_dphi))
                    max_err = max(max_err, err)
            
            if max_err < convergence_threshold:
                convergence_time = t * dt
                break
        
        phase_convergence_time = convergence_time
        
        # 3. 相位稳定性（最后5秒的标准差）
        stable_window = int(5.0 / dt)
        stable_start = max(0, T - stable_window)
        phi_stable = phi_history[stable_start:]
        phase_stability = 1.0 / (1.0 + np.std(np.diff(phi_stable, axis=0)))
        
        # 4. 步态规律性
        phi_diff = np.diff(phi_history, axis=0)
        gait_regularity = 1.0 / (1.0 + np.mean(np.std(phi_diff, axis=0)))
        
        # 5. 步幅一致性（角速度一致性）
        stride_consistency = 1.0 / (1.0 + np.std(omega_history))
        
        # 6. 姿态稳定性
        body_roll_std = np.std(psi_history)
        body_pitch_std = np.std(theta_history)
        body_oscillation = np.sqrt(body_roll_std**2 + body_pitch_std**2)
        
        # 7. 鲁棒性（扰动后恢复）
        if disturbance_time < total_duration:
            disturb_idx = int(disturbance_time / dt)
            post_disturb = phi_history[disturb_idx:]
            
            # 恢复时间
            recovery_time = total_duration - disturbance_time
            for t in range(len(post_disturb)):
                phi_t = post_disturb[t]
                max_err = 0
                for i in range(4):
                    for j in range(i+1, 4):
                        target_dphi = wrap_to_pi(phi_target[j] - phi_target[i])
                        actual_dphi = wrap_to_pi(phi_t[j] - phi_t[i])
                        err = abs(wrap_to_pi(actual_dphi - target_dphi))
                        max_err = max(max_err, err)
                
                if max_err < 0.15:  # 恢复阈值
                    recovery_time = t * dt
                    break
            
            # 偏差（扰动后1秒内的最大偏差）
            deviation_window = int(1.0 / dt)
            deviation_window = min(deviation_window, len(post_disturb))
            deviation_errors = []
            for t in range(deviation_window):
                phi_t = post_disturb[t]
                for i in range(4):
                    for j in range(i+1, 4):
                        target_dphi = wrap_to_pi(phi_target[j] - phi_target[i])
                        actual_dphi = wrap_to_pi(phi_t[j] - phi_t[i])
                        err = abs(wrap_to_pi(actual_dphi - target_dphi))
                        deviation_errors.append(err)
            
            disturbance_deviation = np.max(deviation_errors) if deviation_errors else 0.0
            disturbance_recovery_time = recovery_time
        else:
            disturbance_recovery_time = 0.0
            disturbance_deviation = 0.0
        
        # 8. 频率变化
        frequency_variation = np.std(omega_history)
        
        # 9. 耦合效率
        coupling_efficiency = 1.0 / (1.0 + np.mean(coupling_history))
        
        return SimulationMetrics(
            phase_synchronization=phase_synchronization,
            phase_convergence_time=phase_convergence_time,
            phase_stability=phase_stability,
            gait_regularity=gait_regularity,
            stride_consistency=stride_consistency,
            body_roll_std=body_roll_std,
            body_pitch_std=body_pitch_std,
            body_oscillation=body_oscillation,
            disturbance_recovery_time=disturbance_recovery_time,
            disturbance_deviation=disturbance_deviation,
            frequency_variation=frequency_variation,
            coupling_efficiency=coupling_efficiency,
        )
    
    def run_all_trials(self, n_repeats: int = 3):
        """运行所有消融实验"""
        print("\n" + "="*70)
        print("开始消融实验")
        print("="*70)
        
        all_results = []
        
        for config in self.configs:
            # 多次重复取平均
            trial_results = []
            for i in range(n_repeats):
                print(f"\n  重复 {i+1}/{n_repeats}")
                metrics = self.run_single_trial(config, duration=20.0)
                trial_results.append(metrics)
            
            # 平均指标
            avg_metrics = self._average_metrics(trial_results)
            self.results[config.name] = avg_metrics
            all_results.append({
                'config': config.name,
                **avg_metrics.to_dict()
            })
            
            print(f"\n  平均指标:")
            print(f"    相位同步误差: {avg_metrics.phase_synchronization:.4f} rad")
            print(f"    收敛时间: {avg_metrics.phase_convergence_time:.2f} s")
            print(f"    扰动恢复时间: {avg_metrics.disturbance_recovery_time:.2f} s")
        
        # 保存结果
        df = pd.DataFrame(all_results)
        csv_path = self.output_dir / "ablation_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✅ 结果已保存: {csv_path}")
        
        # 保存JSON
        json_path = self.output_dir / "ablation_results.json"
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        return df
    
    def _average_metrics(self, metrics_list: List[SimulationMetrics]) -> SimulationMetrics:
        """平均多次试验的指标"""
        avg_dict = {}
        for key in metrics_list[0].to_dict().keys():
            values = [getattr(m, key) for m in metrics_list]
            avg_dict[key] = np.mean(values)
        
        return SimulationMetrics(**avg_dict)


# ============ 主执行 ============
if __name__ == '__main__':
    print("="*70)
    print(" 对称AFCO CPG 消融实验")
    print(" Ablation Study for Symmetric AFCO CPG")
    print("="*70)
    
    # 创建执行器
    runner = AblationStudyRunner(output_dir="/mnt/user-data/outputs/ablation_results")
    
    # 运行所有试验
    results_df = runner.run_all_trials(n_repeats=5)
    
    print("\n" + "="*70)
    print("消融实验完成！")
    print("="*70)
    print(results_df.to_string(index=False))
