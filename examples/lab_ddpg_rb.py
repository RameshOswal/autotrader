import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_load.replay_buffer import ReplayBuffer
from data_load.batchifier import Batchifier
from models.ddpg_model import DDPGCritic, DDPGActor, tf, OrnsteinUhlenbeckActionNoise
from literals import ASSET_LIST
import numpy as np
from get_metrics import get_metrics
from sklearn.utils.extmath import softmax




DATA_PATH = "../dataset/Poloneix_Preprocessednew"
BSZ=32
BPTT=10
asset_list=ASSET_LIST
IDX=0
NUM_EPOCHS = 100
INIT_PV=1000
NUM_HID=20
ASSETS = ASSET_LIST
LR_CRITIC = 1e-2
LR_ACTOR = 1e-3
LOG_MINI_BATCH = 5000
randomize_train=False
overlapping_train=True
RB_FACTOR=10
replay=BSZ*RB_FACTOR
GAMMA = 0.99
MIX_FACTOR = 0.001
CLIP_NORM = 5.0
ACTOR_NOISE = OrnsteinUhlenbeckActionNoise(mu=np.zeros(len(ASSETS) + 1))


if __name__ == '__main__':

    batch_gen = Batchifier(data_path=DATA_PATH, bsz=1, bptt=BPTT, idx=IDX,
                           asset_list=ASSETS, randomize_train=randomize_train,
                           overlapping_train=overlapping_train)


    buffer = ReplayBuffer(buffer_size=replay, rewards=True)
    actor = DDPGActor(clip_norm=CLIP_NORM, num_hid=NUM_HID,
                      num_assets=len(ASSETS), bptt=BPTT, lr=LR_ACTOR,
                      mix_factor=MIX_FACTOR)
    critic = DDPGCritic(clip_norm=CLIP_NORM, num_hid=NUM_HID,
                      num_assets=len(ASSETS), bptt=BPTT, lr=LR_CRITIC,
                      mix_factor=MIX_FACTOR)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        critic_losses = []
        actor_rewards = []
        for epoch in range(1,NUM_EPOCHS + 1):
            batch_losses = 0.0
            for bTrainX, bTrainY in batch_gen.load_train():
                alloc_action = actor.predict_allocation(sess, bTrainX) # + ACTOR_NOISE()
                alloc_reward = np.dot(alloc_action.flatten(), bTrainY[:,-1,:].flatten())
                buffer.add(state=bTrainX, action=alloc_action, reward=alloc_reward)
                actor_rewards.append(alloc_reward)
                if buffer.size >= BSZ:
                    market_state_rb, reward_rb, alloc_action_rb = buffer.get_batch(bsz=BSZ)
                    targ_q = critic.target_predict_q(sess, market_state_rb,
                                                     actor.target_predict_allocation(sess, market_state_rb))
                    reward_qrb = targ_q + GAMMA*reward_rb
                    loss_critic = critic.optimize(sess, market_state_rb, alloc_action_rb, reward_qrb)
                    alloc_grads = critic.allocation_grad(sess, market_state_rb,
                                                         actor.predict_allocation(sess, market_state_rb))
                    actor.optimize(sess, market_state_rb, alloc_grads)
                    actor.update_target_network(sess)
                    critic.update_target_network(sess)
                    critic_losses.append(loss_critic)


            print("Epoch {} Average Reward: {}, Critic Loss: {}, validating...".format(epoch, np.mean(actor_rewards),
                                                                                       np.mean(critic_losses)))
            losses = []
            allocation_wts = []
            price_change_vec = []
            for bEvalX, bEvalYFat in batch_gen.load_test():
                bEvalY = bEvalYFat[:,-1,:]
                pred_allocations = actor.predict_allocation(sess, bEvalX)
                assert bEvalY.shape == pred_allocations.shape
                price_change_vec.append(bEvalY)
                allocation_wts.append(pred_allocations)

            true_change_vec = np.concatenate(price_change_vec)
            allocation_wts = np.concatenate(allocation_wts)

            random_alloc_wts = softmax(np.random.random(allocation_wts.shape))
            test_date = "_".join(batch_gen.dp.test_dates[IDX])
            m = get_metrics(dt_range=test_date)
            print("Our Policy:")
            m.apv_multiple_asset(true_change_vec, allocation_wts, get_graph=True, pv_0=INIT_PV, tag="epoch_{}".format(epoch))