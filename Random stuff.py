def draw(screen, elements):
    for element in elements:
        if isinstance(element[0], pygame.Rect):
            pygame.draw.rect(screen, element[1], element[0])
        elif isinstance(element[0], pygame.Surface):
            screen.blit(element[0], element[1])
        elif element[0] == "semicircle":
            draw_semicircle(screen, * element[1:])
        elif element[0] == "line":
            pygame.draw.line(screen, * element[1:])

def draw_semicircle(screen, center, radius, fill_colour, start_angle = 0, border_width = 5, border_colour = "Black", resolution = 5 ):
    points = [center]
    for i in range(180*resolution + 1 ):
        angle = start_angle + i/resolution
        x = center[0] + radius * math.cos(math.radians(angle))
        y = center[1] - radius * math.sin(math.radians(angle))
        points.append((x, y))
    pygame.draw.polygon(screen, fill_colour, points)
    pygame.draw.polygon(screen, border_colour, points, border_width)


    DRAG_COEFFICIENT = 0.4257
    ROLLING_RESISTANCE_COEFFICIENT = 12.8


#
# class Meter:
#     def __init__(self, min_value, max_value):
#         self.min_value = min_value
#         self.max_value = max_value
#         self.value = 0
#         self.range_of_values = self.max_value - self.min_value
#         self.dragging = False
#
#     def update_value(self, value):
#         self.value = resources.clamp_value(value, self.min_value, self.max_value)
#         return (self.value - self.min_value) / self.range_of_values
#
# class HorizontalMeter(Meter):
#     def __init__(self, x, y, width, height, base_colour, knob_colour, min_value, max_value):
#         super().__init__(min_value, max_value)
#         self.knob_width = 10
#         self.base = pygame.Rect(x, y, width, height)
#         self.knob = pygame.Rect((x + (width - self.knob_width) // 2), y, self.knob_width, height)
#         self.elements = ((self.base, base_colour), (self.knob, knob_colour))
#
#     def update_value(self, value):
#         relative_value = super().update_value(value)
#         self.knob.x = relative_value * self.base.width + self.base.left

# class Slider(HorizontalMeter):
#     def event_handle(self, event):
#         if self.knob.collidepoint(event.pos):
#             if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
#                self.dragging = True
#             if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
#                 self.dragging = False
#
#         if self.dragging:
#             click_position = resources.clamp_value(event.pos[0], self.base.left, self.base.right)
#             relative_position = click_position - self.base.left
#
#             percentage_increment = relative_position / (self.base.right - self.base.left)
#             value = round(percentage_increment * self.range_of_values, 2) + self.min_value
#             super().update_value(value)
#
# import numpy as np
# import copy
#
# # ----------------------------
# # Activation function classes
# # ----------------------------
# class Activation:
#     def forward(self, z): raise NotImplementedError
#     def derivative(self, z): raise NotImplementedError  # derivative wrt pre-activation z
#
# class ReLU(Activation):
#     def forward(self, z): return np.maximum(0.0, z)
#     def derivative(self, z): return (z > 0).astype(float)
#
# class Tanh(Activation):
#     def forward(self, z): return np.tanh(z)
#     def derivative(self, z):
#         t = np.tanh(z)
#         return 1.0 - t*t
#
# class Sigmoid(Activation):
#     def forward(self, z):
#         return 1.0 / (1.0 + np.exp(-z))
#     def derivative(self, z):
#         s = self.forward(z)
#         return s * (1.0 - s)
#
# class Identity(Activation):
#     def forward(self, z): return z
#     def derivative(self, z): return np.ones_like(z)
#
# # activation factory
# def get_activation(name):
#     name = name.lower()
#     if name in ('relu',): return ReLU()
#     if name in ('tanh',): return Tanh()
#     if name in ('sigmoid','logistic'): return Sigmoid()
#     if name in ('identity','linear','none'): return Identity()
#     raise ValueError(f"Unknown activation '{name}'")
#
# # ----------------------------
# # Weight initialization helper
# # ----------------------------
# def init_weights(in_dim, out_dim, activation):
#     """
#     - For relu: He normal
#     - For tanh/sigmoid/identity: Glorot (Xavier) uniform
#     Returns shape (in_dim, out_dim)
#     """
#     act_name = activation.__class__.__name__.lower()
#     if 'relu' in act_name:
#         std = np.sqrt(2.0 / in_dim)
#         return np.random.randn(in_dim, out_dim) * std
#     else:
#         limit = np.sqrt(6.0 / (in_dim + out_dim))
#         return np.random.uniform(-limit, limit, size=(in_dim, out_dim))
#
# # ----------------------------
# # Layer class (linear + activation)
# # ----------------------------
# class Layer:
#     def __init__(self, in_dim, out_dim, activation='relu'):
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.activation = get_activation(activation)
#         self.weights = init_weights(in_dim, out_dim, self.activation)  # shape (in, out)
#         self.biases = np.zeros(out_dim)  # shape (out,)
#
#         # caches for backprop
#         self._last_input = None    # shape (batch, in)
#         self._last_z = None        # shape (batch, out)
#         self._last_a = None        # shape (batch, out)
#
#     def forward(self, x):
#         """
#         x: (batch, in_dim)
#         returns a: (batch, out_dim)
#         """
#         z = x.dot(self.weights) + self.biases  # broadcasting biases (out,)
#         a = self.activation.forward(z)
#         # cache
#         self._last_input = x
#         self._last_z = z
#         self._last_a = a
#         return a
#
#     def backward(self, dA):
#         """
#         dA: dL/dA for this layer (batch, out_dim)
#         returns:
#           - dX: dL/dX to pass to previous layer (batch, in_dim)
#           - dW: gradient for weights (in_dim, out_dim)
#           - db: gradient for biases (out_dim,)
#         Notes: we compute average gradient over batch (i.e., divide by batch_size)
#         """
#         batch = dA.shape[0]
#         z = self._last_z
#         x = self._last_input
#         dZ = dA * self.activation.derivative(z)   # (batch, out)
#         dW = x.T.dot(dZ) / batch                   # (in, out)
#         db = dZ.mean(axis=0)                       # (out,)
#         dX = dZ.dot(self.weights.T)                # (batch, in)
#         return dX, dW, db
#
#     def get_params(self):
#         return {'w': self.weights.copy(), 'b': self.biases.copy()}
#
#     def set_params(self, params):
#         self.weights = params['w'].copy()
#         self.biases = params['b'].copy()
#
#
# # ----------------------------
# ----------------------------
# Policy network (Gaussian mean output)
# ----------------------------
class GaussianPolicyNet:
    """
    A simple MLP policy that outputs a mean vector mu(s) for a diagonal Gaussian policy.
    log_std is a separate learnable vector of shape (action_dim,).

    Methods:
      - forward(states): returns mu (B, A)
      - act(states, stochastic=True): sample action(s)
      - gaussian_log_prob(actions, mu): per-sample log-prob
      - gaussian_entropy(): scalar entropy (sum over action dims)
      - policy_loss_and_grads(states, actions, advantages, entropy_coef): compute L and grads
      - apply_gradients(grads, lr, ...): Adam update using grads from policy_loss_and_grads
      - get/set/perturb param vector (for EA)
    """
    def __init__(self, layer_sizes, activations, init_log_std=-1.0, seed=None):
        """
        layer_sizes: list like [obs_dim, h1, h2, ..., action_dim]
        activations: list of activation names for each layer except input (len = len(layer_sizes)-1)
        init_log_std: scalar initial log std
        seed: optional integer seed (local RNG)
        """
        assert len(activations) == len(layer_sizes) - 1
        self.rng = np.random.default_rng(seed)

        # build layers (all Dense)
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activation=activations[i], rng=self.rng))

        # log_std (learnable), shape (action_dim,)
        self.log_std = np.full(layer_sizes[-1], float(init_log_std), dtype=float)

        # Adam optimizer state
        self.adam_state = {
            'mw': [np.zeros_like(l.weights) for l in self.layers],
            'vw': [np.zeros_like(l.weights) for l in self.layers],
            'mb': [np.zeros_like(l.biases)  for l in self.layers],
            'vb': [np.zeros_like(l.biases)  for l in self.layers],
            'mls': np.zeros_like(self.log_std),
            'vls': np.zeros_like(self.log_std),
            't': 0
        }

    # ----------------------------
    # Forward / sampling / logprob helpers
    # ----------------------------
    def forward(self, states):
        """
        states: (B, obs_dim)
        returns mu: (B, action_dim)
        """
        a = states
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def gaussian_log_prob(self, actions, mu):
        """
        actions, mu: (B, A)
        self.log_std: (A,)
        returns: logp per sample (B,)
        Formula:
          log N(a | mu, diag(sigma^2)) = -1/2 * sum( (a-mu)^2 / sigma^2 + 2*log_sigma + log(2π) )
        """
        log_std = self.log_std            # (A,)
        std = np.exp(log_std)             # (A,)
        var = std**2                      # (A,)

        diff = actions - mu               # (B, A)
        # broadcast var and log_std across batch
        logp = -0.5 * np.sum((diff**2) / var + 2.0 * log_std + np.log(2.0 * np.pi), axis=1)  # (B,)
        return logp

    def gaussian_entropy(self):
        """
        Entropy of diagonal Gaussian (sum over action dims).
        H = 0.5 * sum(1 + log(2πσ^2)) = 0.5 * sum(1 + log(2π) + 2*log_std)
        returns scalar
        """
        return 0.5 * np.sum(1.0 + np.log(2.0 * np.pi) + 2.0 * self.log_std)

    def act(self, states, stochastic=True):
        """
        states: (B, obs_dim) or (obs_dim,) for single state
        returns:
          actions: (B, A)
          logp: (B,) per-sample log-prob of chosen actions
          mu: (B, A)
        """
        # ensure 2D batch
        single = False
        if states.ndim == 1:
            states = states[None, :]
            single = True

        mu = self.forward(states)     # (B, A)
        if stochastic:
            eps = self.rng.standard_normal(mu.shape)
            std = np.exp(self.log_std)           # (A,)
            actions = mu + eps * std            # broadcasting -> (B, A)
        else:
            actions = mu

        logp = self.gaussian_log_prob(actions, mu)  # (B,)
        if single:
            return actions[0], logp[0], mu[0]
        return actions, logp, mu

    # ----------------------------
    # Loss + gradient computation (policy gradient, VPG style)
    # ----------------------------
    def policy_loss_and_grads(self, states, actions, advantages, entropy_coef=0.0, normalize_adv=True):
        """
        Compute the policy loss L = -mean(logpi * adv) - entropy_coef * H
        and return gradients wrt every parameter:
          - w_grads: list of arrays matching self.layers' weights
          - b_grads: list of arrays matching self.layers' biases
          - log_std_grad: vector (action_dim,)
        Notes:
          - advantages: (B,) computed outside (returns or advantage estimator)
          - This function does NOT update parameters, it only returns gradients.
        """
        B = states.shape[0]
        if normalize_adv:
            adv_mean = np.mean(advantages)
            adv_std = np.std(advantages) + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        # forward: get mu(s)
        mu = self.forward(states)   # (B, A)

        # scalar losses
        logp = self.gaussian_log_prob(actions, mu)          # (B,)
        loss_policy = -np.mean(logp * advantages)           # scalar
        loss_entropy = -entropy_coef * self.gaussian_entropy()
        total_loss = loss_policy + loss_entropy

        # ---------------------------
        # Gradients wrt distribution parameters
        # ---------------------------
        log_std = self.log_std
        std = np.exp(log_std)
        var = std**2                 # (A,)

        diff = actions - mu          # (B, A)

        # dL/dmu (shape B x A): for loss = -mean(logpi * adv)
        # per-sample gradient: -adv * d logpi/dmu ; d logpi/dmu = (a - mu) / var
        dL_dmu = -(advantages[:, None] * (diff / var))   # (B, A)

        # dL/dlogstd (shape A,): derivative of -mean(logpi * adv) wrt logstd
        # per-sample: d logpi / d logstd = -1 + ((a-mu)^2 / var)
        dlpi_dlogstd = -1.0 + (diff**2) / var            # (B, A)
        dL_dlogstd = -np.mean(advantages[:, None] * dlpi_dlogstd, axis=0)  # (A,)

        # entropy gradient w.r.t log_std: d(-entropy_coef * H)/d log_std = -entropy_coef
        if entropy_coef != 0.0:
            dL_dlogstd += -entropy_coef * np.ones_like(self.log_std)

        # ---------------------------
        # Backpropagate dL/dmu into layers
        # ---------------------------
        w_grads = []
        b_grads = []
        upstream = dL_dmu   # (B, A), gradient at network output
        for layer in reversed(self.layers):
            dW, db, upstream = layer.backward(upstream)
            w_grads.insert(0, dW)
            b_grads.insert(0, db)

        grads = {
            'loss': float(total_loss),
            'w_grads': w_grads,
            'b_grads': b_grads,
            'log_std_grad': dL_dlogstd
        }
        return grads

    # ----------------------------
    # Adam update (applies gradients in-place)
    # ----------------------------
    def apply_gradients(self, w_grads, b_grads, log_std_grad,
                        lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        Applies Adam updates to all weights, biases, and log_std.
        The shapes of w_grads/b_grads must match self.layers.
        """
        st = self.adam_state
        st['t'] += 1
        t = st['t']

        def adam_step(m, v, g):
            m = beta1 * m + (1 - beta1) * g
            v = beta2 * v + (1 - beta2) * (g * g)
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            step = lr * m_hat / (np.sqrt(v_hat) + eps)
            return m, v, step

        # update per-layer
        for i, layer in enumerate(self.layers):
            # weights
            m_w = st['mw'][i]
            v_w = st['vw'][i]
            m_w, v_w, stepW = adam_step(m_w, v_w, w_grads[i])
            layer.weights -= stepW
            st['mw'][i], st['vw'][i] = m_w, v_w

            # biases
            m_b = st['mb'][i]
            v_b = st['vb'][i]
            m_b, v_b, stepB = adam_step(m_b, v_b, b_grads[i])
            layer.biases -= stepB
            st['mb'][i], st['vb'][i] = m_b, v_b

        # log_std
        m_ls = st['mls']
        v_ls = st['vls']
        m_ls, v_ls, stepLS = adam_step(m_ls, v_ls, log_std_grad)
        self.log_std -= stepLS
        st['mls'], st['vls'] = m_ls, v_ls

    # ----------------------------
    # Parameter vector utilities (EA-friendly)
    # ----------------------------
    def get_params_vector(self):
        parts = []
        for l in self.layers:
            parts.append(l.weights.ravel())
            parts.append(l.biases.ravel())
        parts.append(self.log_std.ravel())
        return np.concatenate(parts).astype(float)

    def set_params_vector(self, vec):
        idx = 0
        for l in self.layers:
            n_w = l.input_size * l.output_size
            l.weights = vec[idx: idx + n_w].reshape((l.input_size, l.output_size)).copy()
            idx += n_w
            n_b = l.output_size
            l.biases = vec[idx: idx + n_b].copy()
            idx += n_b
        n_ls = self.log_std.shape[0]
        self.log_std = vec[idx: idx + n_ls].copy()
        idx += n_ls
        if idx != len(vec):
            raise ValueError("Parameter vector size mismatch")

    def perturb_params_vector(self, std, seed=None):
        vec = self.get_params_vector()
        rng = np.random.default_rng(seed) if seed is not None else self.rng
        noise = rng.standard_normal(vec.shape) * std
        return vec + noise

    def add_noise_to_params(self, std, seed=None):
        new_vec = self.perturb_params_vector(std, seed)
        self.set_params_vector(new_vec)








class UIBezierCanvas(UIElement):
    def __init__(self, relative_rect, manager, container=None, enforce_g1=True):
        super().__init__(
            relative_rect=relative_rect,
            manager=manager,
            container=container,
            starting_height=0,
            layer_thickness=1,
            object_id="#bezier_canvas"
        )

        # Data
        self.anchors = []   # list[Vector2]  (anchor points in local coords)
        self.segments = []  # list[dict] with keys: start, c1, c2, end, overridden_c1, overridden_c2

        # Drawing surface sized to the element's relative_rect
        self.image = pygame.Surface((self.relative_rect.width, self.relative_rect.height), pygame.SRCALPHA)

        # Interaction state
        self.dragging = None  # None or dict: {'type':'anchor'/'handle', ...}
        self.anchor_hit_radius = 8
        self.handle_hit_radius = 8
        self.enforce_g1 = enforce_g1

        # Appearance
        self.bg_colour = pygame.Color(40, 40, 40)
        self.border_colour = pygame.Color(200, 200, 200)
        self.curve_colour = pygame.Color(220, 80, 40)
        self.anchor_colour = pygame.Color(40, 200, 80)
        self.handle_colour = pygame.Color(200, 120, 40)
        self.helper_colour = pygame.Color(150, 150, 150)
        self.anchor_radius = 5
        self.handle_radius = 4
        self.curve_width = 3

        # initial image
        self.rebuild()

    # -------------------------
    # Input handling
    # -------------------------
    def process_event(self, event):
        # Convert global mouse pos to local canvas coords helper
        def local_pos_from_event(evpos):
            return Vector2(evpos) - Vector2(self.rect.topleft)

        # Left click down -> either start drag or add anchor
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not self.rect.collidepoint(event.pos):
                return False

            local = local_pos_from_event(event.pos)

            # 1) Hit test anchors (closest)
            for i, a in enumerate(self.anchors):
                if (a - local).length() <= self.anchor_hit_radius:
                    # start dragging anchor i
                    offset = a - local
                    self.dragging = {'type': 'anchor', 'index': i, 'offset': offset}
                    return True

            # 2) Hit test handles (c1/c2) per segment
            for si, seg in enumerate(self.segments):
                if (seg['c1'] - local).length() <= self.handle_hit_radius:
                    offset = seg['c1'] - local
                    seg['overridden_c1'] = True  # mark manual change
                    self.dragging = {'type': 'handle', 'seg_index': si, 'which': 'c1', 'offset': offset}
                    return True
                if (seg['c2'] - local).length() <= self.handle_hit_radius:
                    offset = seg['c2'] - local
                    seg['overridden_c2'] = True
                    self.dragging = {'type': 'handle', 'seg_index': si, 'which': 'c2', 'offset': offset}
                    return True

            # 3) No hit -> add a new anchor (append)
            self.anchors.append(Vector2(local))
            self._generate_segments_from_anchors(preserve_overrides=True)
            self.rebuild()
            return True

        # Left button up -> stop drag
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                # when finishing dragging an anchor, recompute default handles
                if self.dragging['type'] == 'anchor':
                    # anchor moved; regenerate default handles where not overridden
                    self._generate_segments_from_anchors(preserve_overrides=True)
                # end drag
                self.dragging = None
                self.rebuild()
                return True

        # Mouse motion -> if dragging, move the item
        if event.type == pygame.MOUSEMOTION:
            if self.dragging:
                local = Vector2(event.pos) - Vector2(self.rect.topleft)
                d = self.dragging
                if d['type'] == 'anchor':
                    idx = d['index']
                    new_pos = local + d['offset']
                    # clamp to canvas bounds
                    new_pos.x = max(0, min(self.relative_rect.width, new_pos.x))
                    new_pos.y = max(0, min(self.relative_rect.height, new_pos.y))
                    self.anchors[idx] = new_pos
                    # update segment endpoints linked to this anchor
                    self._update_segment_endpoints_from_anchors()
                    # regenerate handles only for non-overridden segments
                    self._generate_segments_from_anchors(preserve_overrides=True)
                    self.rebuild()
                    return True

                elif d['type'] == 'handle':
                    si = d['seg_index']
                    which = d['which']
                    seg = self.segments[si]
                    new_pos = local + d['offset']
                    # clamp
                    new_pos.x = max(0, min(self.relative_rect.width, new_pos.x))
                    new_pos.y = max(0, min(self.relative_rect.height, new_pos.y))
                    seg[which] = new_pos
                    seg[f'overridden_{which}'] = True

                    # If G1 enforcement on, update neighbouring segment's matching handle direction
                    if self.enforce_g1:
                        self._enforce_g1_after_handle_move(seg_index=si, which=which)
                    self.rebuild()
                    return True

        # Right click -> undo last anchor
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            if self.rect.collidepoint(event.pos):
                if self.anchors:
                    self.anchors.pop()
                    self._generate_segments_from_anchors(preserve_overrides=True)
                    self.rebuild()
                return True

        # Keyboard: clear with 'c' or 'C'
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                self.anchors.clear()
                self.segments.clear()
                self.rebuild()
                return True

        return False

    # -------------------------
    # Segment generation & updates
    # -------------------------
    def _generate_segments_from_anchors(self, preserve_overrides=True):
        """
        Build self.segments from anchors using Catmull-Rom -> cubic Bézier conversion.
        If preserve_overrides is True, keep any previously overridden control points for matching indices.
        """
        old = self.segments
        n = len(self.anchors)
        new_segs = []

        for i in range(n - 1):
            p1 = self.anchors[i]        # start
            p2 = self.anchors[i + 1]    # end
            p0 = self.anchors[i - 1] if i - 1 >= 0 else p1  # clamp
            p3 = self.anchors[i + 2] if i + 2 < n else p2  # clamp

            # default handles (Catmull-Rom -> Bezier)
            default_c1 = p1 + (p2 - p0) / 6.0
            default_c2 = p2 - (p3 - p1) / 6.0

            seg = {
                'start': Vector2(p1),
                'c1': Vector2(default_c1),
                'c2': Vector2(default_c2),
                'end': Vector2(p2),
                'overridden_c1': False,
                'overridden_c2': False
            }

            # preserve old overrides/positions for the same segment index
            if preserve_overrides and i < len(old):
                oldseg = old[i]
                if oldseg.get('overridden_c1', False):
                    seg['c1'] = Vector2(oldseg['c1'])
                    seg['overridden_c1'] = True
                if oldseg.get('overridden_c2', False):
                    seg['c2'] = Vector2(oldseg['c2'])
                    seg['overridden_c2'] = True

            new_segs.append(seg)

        self.segments = new_segs

    def _update_segment_endpoints_from_anchors(self):
        """When anchors move, update segment start/end positions to match anchors."""
        for i, seg in enumerate(self.segments):
            if i < len(self.anchors):
                seg['start'] = Vector2(self.anchors[i])
            if i + 1 < len(self.anchors):
                seg['end'] = Vector2(self.anchors[i + 1])

    def _enforce_g1_after_handle_move(self, seg_index, which):
        """
        Keep tangent direction continuous at joins:
        - if which == 'c2', update next segment's c1 to be collinear (opposite direction).
        - if which == 'c1', update previous segment's c2 similarly.
        We try to preserve the neighbor's handle length.
        """
        seg = self.segments[seg_index]
        if which == 'c2':
            # Join at seg.end which is anchor between seg and next_seg
            joint_anchor = seg['end']
            v = seg['c2'] - joint_anchor
            if v.length_squared() == 0:
                return
            # update next seg c1
            next_index = seg_index + 1
            if next_index < len(self.segments):
                next_seg = self.segments[next_index]
                old_vec = next_seg['c1'] - joint_anchor
                old_len = old_vec.length()
                if old_len == 0:
                    old_len = v.length()
                new_c1 = joint_anchor - v.normalize() * old_len
                next_seg['c1'] = new_c1
                next_seg['overridden_c1'] = True

        elif which == 'c1':
            # Join at seg.start; affect previous segment's c2
            joint_anchor = seg['start']
            v = seg['c1'] - joint_anchor
            if v.length_squared() == 0:
                return
            prev_index = seg_index - 1
            if prev_index >= 0:
                prev_seg = self.segments[prev_index]
                old_vec = prev_seg['c2'] - joint_anchor
                old_len = old_vec.length()
                if old_len == 0:
                    old_len = v.length()
                new_c2 = joint_anchor - v.normalize() * old_len
                prev_seg['c2'] = new_c2
                prev_seg['overridden_c2'] = True

    # -------------------------
    # Drawing / rebuild
    # -------------------------
    def rebuild(self):
        """Redraw the committed image (cached)."""
        # clear
        self.image.fill((0, 0, 0, 0))
        # background
        pygame.draw.rect(self.image, self.bg_colour, pygame.Rect(0, 0, self.relative_rect.width, self.relative_rect.height))

        # draw segments
        for seg in self.segments:
            self._draw_cubic_bezier_on_surf(self.image, seg['start'], seg['c1'], seg['c2'], seg['end'], self.curve_colour, self.curve_width)

        # draw helper lines & handles
        for seg in self.segments:
            # helper lines from start->c1 and end->c2
            pygame.draw.line(self.image, self.helper_colour, seg['start'], seg['c1'], 1)
            pygame.draw.line(self.image, self.helper_colour, seg['end'], seg['c2'], 1)
            # handles
            pygame.draw.circle(self.image, self.handle_colour, (int(seg['c1'].x), int(seg['c1'].y)), self.handle_radius)
            pygame.draw.circle(self.image, self.handle_colour, (int(seg['c2'].x), int(seg['c2'].y)), self.handle_radius)

        # draw anchors on top
        for a in self.anchors:
            pygame.draw.circle(self.image, self.anchor_colour, (int(a.x), int(a.y)), self.anchor_radius)

    def _draw_cubic_bezier_on_surf(self, surf, p0, c1, c2, p3, colour, width):
        """Evaluate cubic with De Casteljau (lerp) and draw connected short lines."""
        pts = []
        steps = 48
        for i in range(steps + 1):
            t = i / steps
            a = p0.lerp(c1, t)
            b = c1.lerp(c2, t)
            c = c2.lerp(p3, t)
            d = a.lerp(b, t)
            e = b.lerp(c, t)
            pt = d.lerp(e, t)
            pts.append((int(pt.x), int(pt.y)))
        if len(pts) >= 2:
            pygame.draw.lines(surf, colour, False, pts, width)

    # -------------------------
    # Main draw called each frame
    # -------------------------
    def draw(self, surface):
        # blit cached image
        surface.blit(self.image, self.rect.topleft)
        # border
        pygame.draw.rect(surface, self.border_colour, self.rect, width=1)

        # live preview: if mouse inside and there is at least 1 anchor, preview next segment to mouse
        mx, my = pygame.mouse.get_pos()
        if self.rect.collidepoint((mx, my)) and self.anchors:
            local_mouse = Vector2((mx - self.rect.x, my - self.rect.y))
            last_anchor = self.anchors[-1]

            keys = pygame.key.get_pressed()
            space = keys[pygame.K_SPACE]

            if space:
                # straight preview
                start = Vector2(self.rect.x + last_anchor.x, self.rect.y + last_anchor.y)
                end = Vector2(self.rect.x + local_mouse.x, self.rect.y + local_mouse.y)
                pygame.draw.line(surface, self.curve_colour, start, end, self.curve_width)
            else:
                # compute preview cubic from last anchor -> mouse (use catmull style with clamped neighbours)
                p1 = last_anchor
                p2 = local_mouse
                p0 = self.anchors[-2] if len(self.anchors) >= 2 else p1
                p3 = p2
                c1 = p1 + (p2 - p0) / 6.0
                c2 = p2 - (p3 - p1) / 6.0
                # evaluate and draw
                preview_pts = []
                steps = 30
                for i in range(steps + 1):
                    t = i / steps
                    a = p1.lerp(c1, t)
                    b = c1.lerp(c2, t)
                    c = c2.lerp(p2, t)
                    d = a.lerp(b, t)
                    e = b.lerp(c, t)
                    pt = d.lerp(e, t)
                    preview_pts.append((int(self.rect.x + pt.x), int(self.rect.y + pt.y)))
                if len(preview_pts) >= 2:
                    pygame.draw.lines(surface, self.curve_colour, False, preview_pts, self.curve_width)

        # small on-canvas instructions
        font = pygame.font.SysFont(None, 16)
        info = "LClick: add • Drag anchors/handles • RClick: undo • C: clear • Hold SPACE: straight"
        text_surf = font.render(info, True, (200, 200, 200))
        surface.blit(text_surf, (self.rect.x + 6, self.rect.y + 6))



# import math
# from collections import defaultdict
# import pygame
#
# class SpatialHashGrid:
#     """
#     Spatial hash grid using exact 2D DDA (Amanatides & Woo) to insert every segment into
#     every grid cell it touches. Also supports coloring segments by the last cell they occupy.
#     """
#
#     def __init__(self, cell_size: float):
#         assert cell_size > 0, "cell_size must be > 0"
#         self.cell_size = float(cell_size)
#         self.cells = defaultdict(list)   # (ix,iy) -> [seg_id, ...]
#         self.segments = []               # seg_id -> ((x1,y1),(x2,y2))
#         self.seg_last_cell = {}          # seg_id -> (ix,iy)
#
#     # --------------------
#     # Helpers
#     # --------------------
#     def _cell_coords(self, pt):
#         x, y = pt
#         # floor division works for negatives correctly
#         return int(math.floor(x / self.cell_size)), int(math.floor(y / self.cell_size))
#
#     def _push_to_cell(self, seg_id, cell):
#         """Append seg_id to cell list but avoid duplicate consecutive appends."""
#         lst = self.cells[cell]
#         # DDA won't revisit a cell non-consecutively, so checking last element is fast and safe
#         if not lst or lst[-1] != seg_id:
#             lst.append(seg_id)
#         # remember last cell (for coloring)
#         self.seg_last_cell[seg_id] = cell
#
#     # --------------------
#     # DDA insertion
#     # --------------------
#     def _dda_insert(self, seg_id, p0, p1):
#         """Amanatides & Woo 2D DDA: step through every grid cell touched by segment p0->p1."""
#         x1, y1 = p0
#         x2, y2 = p1
#
#         ix, iy = self._cell_coords(p0)
#         ex, ey = self._cell_coords(p1)
#
#         dx = x2 - x1
#         dy = y2 - y1
#
#         # degenerate: zero-length segment
#         if abs(dx) < 1e-12 and abs(dy) < 1e-12:
#             self._push_to_cell(seg_id, (ix, iy))
#             return
#
#         step_x = 1 if dx > 0 else -1 if dx < 0 else 0
#         step_y = 1 if dy > 0 else -1 if dy < 0 else 0
#
#         # param t: p(t) = p0 + t*(dx,dy), t in [0,1]
#         t_delta_x = (self.cell_size / abs(dx)) if dx != 0 else float('inf')
#         t_delta_y = (self.cell_size / abs(dy)) if dy != 0 else float('inf')
#
#         # first t at which we cross a vertical/horizontal boundary
#         if step_x > 0:
#             next_boundary_x = (ix + 1) * self.cell_size
#             t_max_x = (next_boundary_x - x1) / dx
#         elif step_x < 0:
#             next_boundary_x = (ix) * self.cell_size
#             t_max_x = (next_boundary_x - x1) / dx
#         else:
#             t_max_x = float('inf')
#
#         if step_y > 0:
#             next_boundary_y = (iy + 1) * self.cell_size
#             t_max_y = (next_boundary_y - y1) / dy
#         elif step_y < 0:
#             next_boundary_y = (iy) * self.cell_size
#             t_max_y = (next_boundary_y - y1) / dy
#         else:
#             t_max_y = float('inf')
#
#         # Insert start cell
#         self._push_to_cell(seg_id, (ix, iy))
#
#         # Safety cap for pathological cases (shouldn't hit)
#         max_steps = int(abs(ex - ix) + abs(ey - iy) + 10 + math.ceil(math.hypot(dx, dy) / self.cell_size * 2))
#         steps = 0
#         while (ix != ex or iy != ey) and steps < max_steps:
#             steps += 1
#             if t_max_x < t_max_y:
#                 ix += step_x
#                 t_max_x += t_delta_x
#             else:
#                 iy += step_y
#                 t_max_y += t_delta_y
#             self._push_to_cell(seg_id, (ix, iy))
#
#     # --------------------
#     # API
#     # --------------------
#     def add_segment(self, segment):
#         """
#         Insert a segment ((x1,y1),(x2,y2)) into the grid.
#         The segment is stored by ID and DDA-walked into cells.
#         """
#         seg_id = len(self.segments)
#         self.segments.append(segment)
#         self._dda_insert(seg_id, segment[0], segment[1])
#
#     def build_from_segments(self, seg_list):
#         """Clear and bulk insert."""
#         self.cells.clear()
#         self.segments = []
#         self.seg_last_cell.clear()
#         for seg in seg_list:
#             # validate shape
#             assert isinstance(seg, tuple) and len(seg) == 2, "segment must be ((x1,y1),(x2,y2))"
#             self.add_segment(seg)
#
#     def get_cell_segments(self, point):
#         """Return segment tuples stored in the cell containing point."""
#         cell = self._cell_coords(point)
#         return [self.segments[sid] for sid in self.cells.get(cell, [])]
#
#     # --------------------
#     # Debug / visualization
#     # --------------------
#     def _color_from_cell(self, cell):
#         ix, iy = cell
#         key = (ix * 73856093) ^ (iy * 19349663)
#         key &= 0xFFFFFF
#         return ((key >> 16) & 0xFF, (key >> 8) & 0xFF, key & 0xFF)
#
#     def get_segment_color(self, seg_id):
#         cell = self.seg_last_cell.get(seg_id)
#         return self._color_from_cell(cell) if cell is not None else (0, 0, 0)
#
#     def draw_colored_segments(self, surface, width=2):
#         """Draw each segment colored by its last cell assignment."""
#         for sid, seg in enumerate(self.segments):
#             (x1, y1), (x2, y2) = seg
#             pygame.draw.line(surface, self.get_segment_color(sid), (x1, y1), (x2, y2), width)
#
#     def draw_grid(self, surface, outline_color=(80,80,80), max_cells=20000):
#         """Draw outlines of occupied cells (limit to avoid huge draws)."""
#         count = 0
#         for (ix, iy), lst in self.cells.items():
#             if count >= max_cells:
#                 break
#             r = pygame.Rect(ix * self.cell_size, iy * self.cell_size, self.cell_size, self.cell_size)
#             pygame.draw.rect(surface, outline_color, r, 1)
#             count += 1
#
#     # --------------------
#     # Utility: auto cell size
#     # --------------------
#     @staticmethod
#     def suggest_cell_size_from_segments(seg_list, multiplier=2.0, min_size=6, max_size=64):
#         """
#         Compute mean segment length and choose cell_size = clamp(mean_len * multiplier, min_size, max_size).
#         multiplier=2 is a good starting value.
#         """
#         if not seg_list:
#             return min_size
#         total = 0.0
#         for (x1,y1),(x2,y2) in seg_list:
#             total += math.hypot(x2-x1, y2-y1)
#         mean_len = total / len(seg_list)
#         cell = int(max(min_size, min(max_size, round(mean_len * multiplier))))
#         return cell
#
#     # --------------------
#     # Debug verifier (optional)
#     # --------------------
#     def verify_segment_cells(self, seg_id, samples_per_cell=3):
#         """Brute force sample along the segment and compare DDA cells vs sampled cells for debugging."""
#         (x1,y1),(x2,y2) = self.segments[seg_id]
#         length = math.hypot(x2-x1, y2-y1)
#         if length == 0:
#             sampled = { self._cell_coords((x1,y1)) }
#         else:
#             samples = max(2, int(length / (self.cell_size / samples_per_cell)))
#             sampled = set()
#             for i in range(samples+1):
#                 t = i / samples
#                 x = x1 + (x2-x1) * t
#                 y = y1 + (y2-y1) * t
#                 sampled.add(self._cell_coords((x,y)))
#         dda_cells = { cell for cell,lst in self.cells.items() if seg_id in lst }
#         missing = sampled - dda_cells
#         extra = dda_cells - sampled
#         return dda_cells, sampled, missing, extra
