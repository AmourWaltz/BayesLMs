import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import pdb

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


def _clip(grads, max_norm):
    total_norm = 0
    for g in grads:
        param_norm = g.data.norm(2)
        total_norm += param_norm ** 2
    total_norm = total_norm ** 0.5
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for g in grads:
            g.data.mul_(clip_coef)
    return clip_coef


class Architect(object):

  def __init__(self, model, ntokens, args):
    self.network_weight_decay = args.wdecay
    self.network_clip = args.clip
    self.model = model
    self.ntokens = ntokens
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(), lr=args.arch_lr, weight_decay=args.arch_wdecay)

  def _compute_unrolled_model(self, data, target, eta):
    output = self.model.forward(data)
    loss = self.criterion(output.view(-1, self.ntokens), target)
    theta = _concat(self.model.parameters()).data
    grads = torch.autograd.grad(loss, self.model.parameters())
    clip_coef = _clip(grads, self.network_clip)
    dtheta = _concat(grads).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, dtheta))
    return unrolled_model, clip_coef

  def step(self,
          input_train, target_train,
          input_valid, target_valid,
          network_optimizer, unrolled):
    eta = network_optimizer.param_groups[0]['lr']
    # pdb.set_trace()
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input, target):
    output = self.model.forward(input)
    loss = self.criterion(output.view(-1, self.ntokens), target)
    loss.backward()

  def _backward_step_unrolled(self,
          input_train, target_train,
          input_valid, target_valid, eta):
    # pdb.set_trace()
    unrolled_model, clip_coef = self._compute_unrolled_model(input_train, target_train, eta)
    # unrolled_loss, hidden_next = unrolled_model._loss(hidden_valid, input_valid, target_valid)
    output = unrolled_model.forward(input_valid)
    unrolled_loss = self.criterion(output.view(-1, self.ntokens), target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    dtheta = [v.grad for v in unrolled_model.parameters()]
    _clip(dtheta, self.network_clip)
    vector = [dt.data for dt in dtheta]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train, r=1e-2)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta * clip_coef, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)
    # return hidden_next

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    # loss, _ = self.model._loss(hidden, input, target)
    output = self.model.forward(input)
    loss = self.criterion(output.view(-1, self.ntokens), target)

    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    # loss, _ = self.model._loss(hidden, input, target)
    output = self.model.forward(input)
    loss = self.criterion(output.view(-1, self.ntokens), target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

