package it.unibo.casestudy

import it.unibo.alchemist.model.implementations.molecules.SimpleMolecule
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.casestudy.Model.TasksPerSolver

object Model {
  type SkillSet = Set[String]
  type FeedbackSet = Set[Feedback]
  type SolverSet = Set[Solver]
  type SolverFitnessSet = Map[Solver,Double]
  type ProblemSet = Set[Problem]
  type RoleSet = Set[Role]
  type ProblemsDetails = Map[Problem,FeedbackSet]
  type TaskAssignment = Map[Problem,SolverFitnessSet]
  type TasksPerSolver = Map[Solver,ProblemSet]

  implicit def intToRole(x: Int): Role = x match {
    case 0 => Robot
    case 1 => Technician
    case 2 => Coordinator
  }
}

import it.unibo.casestudy.Model._

case class Problem(problemID: String, occurrenceID: String)(val description: String = "", val reportedBy: Option[Int] = None, val timestamp: Long = 0) {
  override def toString: String = s"Problem(id=$problemID, occ=$occurrenceID)($description; reported by=$reportedBy; $timestamp)"
}

case class Solver(device: ID)(val role: Role, val details: String, val available: Boolean, val skills: SkillSet){
  def setAvailable(updatedAvailability: Boolean = true) = Solver(device)(role, details, updatedAvailability, skills)

  override def toString: String =
    s"Solver(dev=$device)(\n\t$role;\n\t$details;\n\tavailable=$available;\n\tskills=$skills)\n"
}
case class Feedback(by: ID, problem: Problem)(val observations: Set[FeedbackObservation] = Set()){
  override def toString: String = s"Feedback(by=$by; problem=$problem)($observations)"

  def +(fo: FeedbackObservation) = Feedback(by,problem)(observations+fo)
}

object TaskAssignment {
  def empty: TaskAssignment = Map.empty
}
object TasksPerSolver {
  def empty: TasksPerSolver = Map.empty
}

case class ProblemClasses(unassigned: ProblemsDetails = Map(),
                          assigned: ProblemsDetails = Map(),
                          finished: ProblemsDetails = Map())

trait FeedbackObservation
case object Accepted extends FeedbackObservation
case object Completed extends FeedbackObservation
case class Progress(percent: StrictPercent) extends FeedbackObservation
case class MoreResourcesNeeded(roles: RoleSet) extends FeedbackObservation

case class StrictPercent(percent: Double){ assert(percent>=0 && percent <=1) }

trait Role
case object Coordinator extends Role
case object Technician extends Role
case object Robot extends Role

class CaseStudy extends AggregateProgram
  with StandardSensors with ScafiAlchemistSupport with FieldUtils with ExplicitFields with BuildingBlocks {

  // Simulation parameters
  val grainParameter = "leaderElectionGrain"
  val startProblemDetectionParameter = "startProblemDetectionParameter"
  val endProblemDetectionParameter = "endProblemDetectionParameter"

  // Local molecules
  val roleProperty = "role"
  val feedbackProperty = "feedback"
  val localProblemsSensor = "localProblems"
  val localSolverSensor = "solverProfile"
  val strongSkillPlaceholder = "strong"

  val problemsUnassigned = "problemsUnassigned"
  val problemsAssigned = "problemsAssigned"
  val problemsFinished = "problemsFinished"

  // Global molecules
  val problemsFound = "problemsFound"
  val problemsCompleted = "problemsCompleted"
  val globalProblems = "globalProblems"
  val globalProblemsCompleted = "globalProblemsCompleted"

  def globalRead[T](s: String): T = alchemistEnvironment.getNodeByID(0).getContents.get(new SimpleMolecule(s)).asInstanceOf[T]
  def globalWrite[T](s: String, v: T) = alchemistEnvironment.getNodeByID(0).setConcentration(new SimpleMolecule(s), v)

  lazy val initialisation = {
    globalWrite(problemsFound, 0)
    globalWrite(globalProblems, Set.empty[Problem])
    globalWrite(problemsCompleted, 0)
    globalWrite(globalProblemsCompleted, Set.empty[Problem])

    node.put(feedbackProperty, Set())
    node.put(localProblemsSensor, Set.empty[Problem])
    node.put(localSolverSensor, Solver(mid)(role, s"Details for $mid", available = nextRandom()>0.1,
      skills = Set("a","b","c") ++ (if(nextRandom()>0.9) Set(strongSkillPlaceholder) else Set.empty[String])
    ))
  }
  var k = 0L

  def perRoundPreparation = {
    k = rep(0)(_+1)

    node.put(problemsUnassigned, 0)
    node.put(problemsAssigned, 0)
    node.put(problemsFinished, 0)

    if(k>startProblemDetection & k<endProblemDetection & k%25==0 & nextRandom()>0.98){
      val p = Problem(if(nextRandom()>0.7) "a" else if(nextRandom()>0.4) "b" else "c", mid+"-"+k)("someDescription", Some(mid), timestamp = timestamp())
      node.put(localProblemsSensor, node.get[Set[Problem]](localProblemsSensor)+p)

      globalWrite(globalProblems, globalRead[Set[Problem]](globalProblems)+p)
    }

    if(mid==0) {
      val nProbFound = globalRead[Set[Problem]](globalProblems).size
      val nProbCompl = globalRead[Set[Problem]](globalProblemsCompleted).size
      globalWrite[Int](problemsFound, nProbFound)
      globalWrite[Int](problemsCompleted, nProbCompl)
    }
  }

  def startProblemDetection: Int = node.get(startProblemDetectionParameter)
  def endProblemDetection: Int = node.get(endProblemDetectionParameter)
  def grain: Double = node.get(grainParameter)
  def infoPropagationNetwork: Boolean = true // might exclude some devices (just make sure nbrhood allows connectedness)
  def role: Role = node.get[Int](roleProperty)
  def priorityField: Double = if(role==Coordinator) -10 else 10
  def problemOccurrences: ProblemSet = node.get[ProblemSet](localProblemsSensor)
  def solverProfile: Solver = node.get[Solver](localSolverSensor)
  def updateSolverProfile(s: Solver) = node.put[Solver](localSolverSensor, s)

  def feedback: FeedbackSet =
    node.get[FeedbackSet](feedbackProperty)

  override type MainResult = Any
  override def main = {
    initialisation
    perRoundPreparation

    val metric: Metric = () => nbrRange()

    val coordinators = branch(!(k>=300 && k<=310 && role==Coordinator)){ priorityS(grain, metric, priorityField) }{ false }

    // Builds up the collection/distribution hop-by-hop network
    val potential = branch(infoPropagationNetwork){ distanceTo(coordinators) }{ Double.PositiveInfinity }

    // Problems reported by workers, area-wise
    val problems = collectSets(downTo = potential, problemOccurrences)

    // Profiles of workers + feedback about problems managed by them
    val solvers = collectSet(downTo = potential, solverProfile)

    // Feedbacks about problems by workers
    val feedbacks = collectSets(downTo = potential, feedback).groupBy(_.problem)

    // Assignment of problems to solvers, by coordinators
    val tasks = branch(coordinators) {
      // Organise problems into sets: unassigned, assigned, finished
      val problemClasses = organizeProblems(problems, solvers, feedbacks)
      node.put("1problems", problemClasses)
      node.put(problemsUnassigned, problemClasses.unassigned.size)
      node.put(problemsAssigned, problemClasses.assigned.size)
      node.put(problemsFinished, problemClasses.finished.size)

      // Allocate is a local optimisation function
      val allocate: (Boolean, ProblemClasses, SolverSet) => TasksPerSolver = if (node.get[Double]("smartness") == 1.0) {
        smartAllocation
      } else {
        naiveAllocation
      }
      allocate(coordinators, problemClasses, solvers)
    } {
      TasksPerSolver.empty
    }

    // Distribution of assignments
    val assignments = broadcast(potential, tasks, metric)

    // Execution of assignments
    execute(assignments)

    // LOGGING
    node.put("0solvers", solvers)
    node.put("3tasksPerSolvers", tasks)
    node.put("4assignments", assignments)

    node.put("coordinator", coordinators)
    node.put("potential", potential)
  }

  def organizeProblems(potentiallyNewProblems: ProblemSet,
                       currentSolvers: SolverSet,
                       feedbacks: ProblemsDetails): ProblemClasses =
    rep[(ProblemsDetails,ProblemClasses)]((Map(),ProblemClasses())) { case (pset, pc) =>
      // Add newly found problems
      val allProblems: ProblemsDetails = pset ++ (potentiallyNewProblems--pset.keys).map(_ -> Set.empty[Feedback]).toMap
      // Merge feedback into
      val allProblemsWithInfo: ProblemsDetails = mergeFeedback(allProblems, feedbacks)

      var finished: ProblemsDetails = allProblemsWithInfo.filter(_._2.exists(_.observations.exists(_==Completed)))
      globalWrite(globalProblemsCompleted, globalRead[Set[Problem]](globalProblemsCompleted) ++ finished.keySet)
      var assigned: ProblemsDetails = (allProblemsWithInfo -- finished.keySet).filter(_._2.exists(_.observations.exists(_==Accepted)))
      var unassigned: ProblemsDetails = (allProblemsWithInfo -- finished.keySet) -- assigned.keySet

      assert(allProblemsWithInfo.size==finished.size+assigned.size+unassigned.size)

      (allProblems, ProblemClasses(unassigned, assigned, finished))
    }._2

  def mergeFeedback(pd1: ProblemsDetails, pd2: ProblemsDetails) = {
    pd1++pd2++(pd1.keySet.intersect(pd2.keySet)).map(p => p -> {
      pd1(p) ++ pd2(p)
    })
  }

  def execute(assignments: TasksPerSolver) = {
    updateSolverProfile(solverProfile.setAvailable(true))

    val execRandomness = nextRandom()

    val localProblems = rep[ProblemSet](Set.empty){ case ps =>
      val assigned = assignments.getOrElse(solverProfile, Set.empty)
      val newPs = ps union assigned
      node.put("assigned_problems", newPs)
      updateSolverProfile(solverProfile.setAvailable(true))
      node.put("assignee", 0)
      newPs.filter(p => {
        val fbs = node.get[FeedbackSet](feedbackProperty)
        val fb = fbs.find(_.problem==p).getOrElse(Feedback(mid,p)(Set.empty))
        if(!fb.observations.contains(Completed)) {
          updateSolverProfile(solverProfile.setAvailable(false))
          node.put("assignee", 1)
        }
        // Filter out only those which are not assigned to me anymore (i.e., those for which the 'finished' feedback has reached the coordinator)
        newPs.contains(p) | (if(fb.observations.isEmpty) {
          node.put(feedbackProperty, fbs - fb + (fb + Accepted))
          true
        } else if(fb.observations.size==1 &&
          timestamp()-p.timestamp+(if(solverProfile.skills.contains(strongSkillPlaceholder)) 200 else 0) > (200+300*execRandomness).toLong) {
          node.put(feedbackProperty, fbs - fb + (fb + Completed))
          false
        } else false)
      })
    }
  }

  // TODO: should also consider the vicinity of solvers to problems (so, possibly extend data with location info)
  def allocation(coordinators: Boolean, problemClasses: ProblemClasses, solvers: SolverSet, fitness: (Solver, Problem)=>Double): TasksPerSolver = {
    val problems = problemClasses.unassigned
    var availableSolvers = solvers.filter(_.available)
    //assert(availableSolvers.size>0)

    val potentialAssignments: TaskAssignment = // map from Problems to SolverSets
      problems.keySet.map(p => p -> availableSolvers.map(s => s -> fitness(s, p)).filter(_._2 > 0).toMap).toMap

    var tasksPerSolver = TasksPerSolver.empty
    for ((p:(Problem,SolverFitnessSet)) <- potentialAssignments) {
      if (!p._2.isEmpty) {
        val suitableSolvers = p._2.toList.sortBy(-_._2)
        val assignee = suitableSolvers.head._1
        availableSolvers -= assignee
        tasksPerSolver += assignee -> (tasksPerSolver.getOrElse(assignee, Set.empty)+p._1)

        // second solver for the task
        suitableSolvers.tail.headOption.foreach(s => tasksPerSolver += s._1 -> (tasksPerSolver.getOrElse(s._1, Set.empty)+p._1))
      }
    }
    tasksPerSolver
  }

  def naiveAllocation(coordinators: Boolean, problemClasses: ProblemClasses, solvers: SolverSet): TasksPerSolver =
    allocation(coordinators, problemClasses, solvers, naiveFitness)

  def smartAllocation(coordinators: Boolean, problemClasses: ProblemClasses, solvers: SolverSet): TasksPerSolver =
    allocation(coordinators, problemClasses, solvers, smartFitness)

  def naiveFitness(solver: Solver, problem: Problem): Double =
    if(!solver.available) 0 else if(solver.skills.contains(problem.problemID)) 1 else 0

  def smartFitness(solver: Solver, problem: Problem): Double =
    if(!solver.available) 0 else if(solver.skills.contains(problem.problemID)) {
      if(solver.skills.contains(strongSkillPlaceholder)) 1 else 0.5
    } else 0

  def collectMap[K,V](downTo: Double, local: Map[K,V], merge: (K,V,V)=>V) =
    C[Double, Map[K,V]](downTo, (m1,m2) => {
      (m1 ++ m2) ++ (m1.keySet.intersect(m2.keySet)).map(k => k -> merge(k,m1(k),m2(k)))
    }, local, Map.empty)

  def collectSets[T](downTo: Double, local: Set[T]) =
    C[Double, Set[T]](downTo, _.union(_), local, Set.empty)

  def collectSet[T](downTo: Double, local: T) =
    collectSets(downTo, Set(local))

  def priorityS(grain: Double, metric: Metric, prioField: Double) =
    breakUsingUids(randomUid(prioField), grain, metric, prioField)

  def randomUid(prioField: Double): (Double, ID) = rep((nextRandom()*prioField, mid())) { v => (v._1, mid()) }

  def distTo(source: Boolean, metric: () => Double = nbrRange) =  classicGradient(source, metric)

  def breakUsingUids(uid: (Double, ID), grain: Double, metric: Metric,  prioField: Double): Boolean =
    uid == rep(uid) { lead: (Double, ID) =>
      distanceCompetition(distTo(uid == lead, metric), lead, uid, grain, metric, prioField)
    }

  def distanceCompetition(d: Double, lead: (Double, ID), uid: (Double, ID),  grain: Double,  metric: Metric,  prioField: Double): (Double, ID) = {
    val inf: (Double, ID) = (Double.PositiveInfinity, uid._2)
    mux(d > grain & prioField<0) {
      uid
    } {
      mux(d >= (0.5 * grain)) {
        inf
      } {
        minHood {
          mux(nbr(d) + metric() >= 0.5 * grain) { nbr(inf) } { nbr(lead) }
  } } } }

  // LIBRARY FUNCTIONS ADAPTED

  def broadcast[V](potential: Double, field: V, metric: Metric): V =
    G_along[V](potential, metric, field, (v: V) => v)
}