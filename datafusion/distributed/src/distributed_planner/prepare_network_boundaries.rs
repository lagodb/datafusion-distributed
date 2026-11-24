use crate::common::TreeNodeExt;
use crate::stage::LocalStage;
use crate::{DistributedLeafExec, NetworkBoundaryExt, NetworkCoalesceExec, Stage};
use datafusion::common::Result;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::physical_plan::ExecutionPlan;
use std::sync::Arc;
use uuid::Uuid;

/// Prepares every [NetworkBoundary] in the plan for distributed execution:
/// - Elides ones whose producer and consumer sides both run on a single task
/// - Scales the producer-stage head of the survivors to feed all consumer tasks
/// - Stamps each surviving stage with a unique `(query_id, num)` identifier.
/// - Moves a head stage holding a leaf whose `original` only a worker can execute into a
///   single-task stage of its own (see [head_needs_worker]).
pub(crate) fn prepare_network_boundaries(
    plan: Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut stage_id = 1;
    let query_id = Uuid::new_v4();

    let transformed = plan.transform_up_with_task_count(1, |plan, task_count| {
        let Some(nb) = plan.as_network_boundary() else {
            return Ok(Transformed::no(plan));
        };
        // If the input stage is already remote, it was already sent over the network, so nothing else
        // we can do here.
        let Stage::Local(input_stage) = nb.input_stage() else {
            return Ok(Transformed::no(plan));
        };

        // 1) If there are both 1 producer and consumer tasks, optimize the network boundary out.
        if task_count == 1 && input_stage.tasks == 1 {
            return Ok(Transformed::yes(Arc::clone(&input_stage.plan)));
        }

        // 2) Scale up the head node of the input stage in order to account for the amount of partition
        //    and consumer count above it.
        let plan = nb
            .producer_head(task_count)?
            .insert(Arc::clone(&input_stage.plan))?;

        // 3) Make sure the input stage can be uniquely identified with a stage index and query id.
        //    If there were already some `query_id` and `num` that's fine.
        let nb = nb.with_input_stage(Stage::Local(LocalStage {
            query_id,
            num: stage_id,
            plan,
            tasks: input_stage.tasks,
            metrics_set: Default::default(),
        }))?;
        stage_id += 1;
        Ok(Transformed::yes(nb))
    })?;

    let plan = transformed.data;
    if !head_needs_worker(&plan)? {
        return Ok(plan);
    }
    // This follows the elision above, which would remove the new 1-to-1 boundary again. A
    // single-task boundary keeps the head's output partitioning.
    let input_properties = Arc::clone(plan.properties());
    let stage = Stage::Local(LocalStage {
        query_id,
        num: stage_id,
        plan,
        tasks: 1,
        metrics_set: Default::default(),
    });
    Ok(Arc::new(NetworkCoalesceExec::try_from_stage(
        stage,
        input_properties,
        1,
    )?))
}

/// Whether the head stage, which the coordinator runs and where every [DistributedLeafExec]
/// executes its `original`, holds a leaf whose `original` only a worker can execute.
///
/// The head holds such a leaf either because a 1-to-1 boundary above its stage was elided or
/// because no boundary was ever injected above it. A plan without any boundary is not
/// distributed, so its head stays as it is.
///
/// Only static planning reaches here: with a dynamic task count, boundaries are injected at
/// execution time and the head is not checked.
fn head_needs_worker(plan: &Arc<dyn ExecutionPlan>) -> Result<bool> {
    let mut distributed = false;
    let mut worker_only = false;
    plan.apply(|node| {
        if node.is_network_boundary() {
            distributed = true;
            return Ok(TreeNodeRecursion::Jump);
        }
        if let Some(leaf) = node.downcast_ref::<DistributedLeafExec>() {
            worker_only |= !leaf.coordinator_executes_original();
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(distributed && worker_only)
}
