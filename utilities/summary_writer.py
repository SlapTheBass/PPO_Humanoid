from tensorboardX import SummaryWriter
import torch


def AddStatistics(writer, advantage, values, loss_policy, loss_value, count_steps, step_idx):

    writer.add_scalar("advantage", advantage.mean().item(), step_idx)
    writer.add_scalar("values", values.mean().item(), step_idx)
    writer.add_scalar("loss_policy", loss_policy / count_steps, step_idx)
    writer.add_scalar("loss_value", loss_value / count_steps, step_idx)