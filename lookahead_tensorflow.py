from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import optimizer


class Lookahead(optimizer.Optimizer):
    '''Tensorflow implementation of the lookahead wrapper.

    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    '''

    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, use_locking=False, name="Lookahead"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        """
        super(Lookahead, self).__init__(use_locking, name)
        self.optimizer = optimizer
        self._la_step = 0
        self._la_alpha = la_alpha
        self._total_la_steps = la_steps

    def _create_slots(self, var_list):
        self.optimizer._create_slots(var_list)

        self._var_list = var_list
        first_var = min(var_list, key=lambda x: x.name)
        self._create_non_slot_variable(initial_value=self._la_step,
                                       name="la_step",
                                       colocate_with=first_var)

        # Create slots for the cached parameters.
        for v in var_list:
            self._zeros_slot(v, "cached_params", self._name)

    def _prepare(self):
        self.optimizer._prepare()

        la_alpha = self._call_if_callable(self._la_alpha)
        total_la_steps = self._call_if_callable(self._total_la_steps)

        self._la_alpha_t = ops.convert_to_tensor(la_alpha, name="la_alpha")
        self._total_la_steps_t = ops.convert_to_tensor(total_la_steps, name="total_la_steps")

    def _get_la_step_accumulators(self):
        with ops.init_scope():
            if context.executing_eagerly():
                graph = None
            else:
                graph = ops.get_default_graph()
            return self._get_non_slot_variable("la_step", graph=graph)

    def _apply_dense(self, grad, var):
        return self.optimizer._apply_dense(grad, var)

    def _resource_apply_dense(self, grad, var):
        return self.optimizer._resource_apply_dense(grad, var)

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        return self.optimizer._apply_sparse_shared(grad, var, indices, scatter_add)

    def _apply_sparse(self, grad, var):
        return self.optimizer._apply_sparse(grad, var)

    def _resource_scatter_add(self, x, i, v):
        return self.optimizer._resource_scatter_add(x, i, v)

    def _resource_apply_sparse(self, grad, var, indices):
        return self.optimizer._resource_apply_sparse(grad, var, indices)

    def _finish(self, update_ops, name_scope):
        inner_finish_op = self.optimizer._finish(update_ops, name_scope)

        with ops.control_dependencies([inner_finish_op, ]):
            la_step = self._get_la_step_accumulators()
            with ops.colocate_with(la_step):
                def update_la_step_func():
                    # update the la_step
                    return control_flow_ops.group([la_step.assign(
                        la_step + 1, use_locking=self._use_locking), ])

                def pull_back_func():
                    # update the la_step
                    update_la_step = la_step.assign(
                        0, use_locking=self._use_locking)
                    # interpolate the variables
                    interpolation = [v.assign(
                        self.get_slot(v, "cached_params") + self._la_alpha_t * (v - self.get_slot(v, "cached_params")))
                                     for v in self._var_list]

                    # update the cached params
                    with ops.control_dependencies(interpolation):
                        update_cached_params = [self.get_slot(v, "cached_params").assign(updated_v) for v, updated_v in
                                                zip(self._var_list, interpolation)]
                    return control_flow_ops.group([update_la_step, ] + interpolation + update_cached_params)

                # condition for when to pull back the params
                condition = tf.greater_equal(la_step, self._total_la_steps_t)
                update_lookahead_states = tf.cond(condition,
                                                  pull_back_func,
                                                  update_la_step_func,
                                                  )

        return control_flow_ops.group([inner_finish_op, update_lookahead_states],
                                      name=name_scope)

    def _call_if_callable(self, param):
        """Call the function if param is callable."""
        return param() if callable(param) else param
