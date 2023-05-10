package org.jgrapht.alg.tour;

import org.javatuples.Pair;
import org.jgrapht.Graph;
import org.jgrapht.GraphPath;
import org.jgrapht.Graphs;
import org.jgrapht.alg.cycle.CycleDetector;
import org.jgrapht.alg.interfaces.HamiltonianCycleAlgorithm;
import org.jgrapht.alg.interfaces.HamiltonianCycleImprovementAlgorithm;
import org.jgrapht.alg.tour.HamiltonianCycleAlgorithmBase;
import org.jgrapht.graph.GraphWalk;
import org.jgrapht.graph.MaskSubgraph;
import org.jgrapht.graph.SimpleDirectedGraph;
import org.jgrapht.traverse.DepthFirstIterator;

import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;


public class LinKernighanHeuristicTSP<V, E> extends
        HamiltonianCycleAlgorithmBase<V, E>
        implements
        HamiltonianCycleImprovementAlgorithm<V, E> {
    private final int passes;
    private final HamiltonianCycleAlgorithm<V, E> initializer;

    private final boolean debugPrinting;

    public LinKernighanHeuristicTSP(int maximumPasses, HamiltonianCycleAlgorithm<V, E> initializer, boolean debugPrinting) {
        this.passes = maximumPasses;
        this.initializer = initializer;
        this.debugPrinting = debugPrinting;
    }

    @Override
    public GraphPath<V, E> getTour(Graph<V, E> graph) {
        return improveTour(initializer.getTour(graph));
    }

    @Override
    public GraphPath<V, E> improveTour(GraphPath<V, E> tour) {
        if (debugPrinting) System.out.println("LIN-KERNIGHAN TOUR IMPROVEMENT");

        for (int i = 0; i < passes; i++) {

            if (debugPrinting) System.out.println("Iteration: " + (i + 1));

            GraphPath<V, E> improved = improveTourSingleIteration(tour);

            if (improved.equals(tour)) {
                if (debugPrinting) System.out.println("Abort calculation, no changes were made in this iteration.");
                break;
            }

            if (debugPrinting)
                System.out.println("Iteration finished with gains of: " + (tour.getWeight() - improved.getWeight()));

            tour = improved;
        }
        return tour;
    }

    private GraphPath<V, E> improveTourSingleIteration(final GraphPath<V, E> initialTour) {

        GraphPath<V, E> curOpt = initialTour;
        GraphPath<V, E> tour = initialTour;
        final Set<V> verticesSet = new HashSet<>(tour.getVertexList());

        for (V v : verticesSet) {

            Set<V> used = new HashSet<>();
            used.add(v);

            final List<E> edgeList = tour.getEdgeList();
            V p = tour.getGraph().edgesOf(v).stream()
                    .filter(edgeList::contains)
                    .map(e -> Graphs.getOppositeVertex(initialTour.getGraph(), e, v))
                    .findFirst()
                    .orElseThrow(() -> new RuntimeException("Could not retrieve predecessor of v=" + v));

            Optional<V> r_optional = findShortcut(p, v, tour, used); //calculate shortcut, if possible

            while (r_optional.isPresent()) {

                V r = r_optional.get();
                used.add(r);

                GraphWalk<V, E> deltaPath = constructDeltaPath(tour, p, v, r);
                Pair<GraphPath<V, E>, V> newTour_t = transformDeltaPathToTour(deltaPath, p, v, r);

                tour = newTour_t.getValue0();

                if (tour.getWeight() < curOpt.getWeight()) {
                    if (debugPrinting)
                        System.out.println(debugPrint(curOpt.getWeight() - tour.getWeight(), p, v, r, newTour_t.getValue1()));
                    curOpt = tour; //save new best
                }

                p = newTour_t.getValue1();

                r_optional = findShortcut(p, v, tour, used);

            }

            tour = curOpt;
        }

        return curOpt;
    }

    private String debugPrint(double saved, V p, V v, V r, V t) {
        return String.format("Saved %f by shortcutting %s to %s. %s was removed, while %s was added.",
                saved,
                ("(p,v)=(" + p + "," + v + ")"),
                ("(p,r)=(" + p + "," + r + ")"),
                ("(r,t)=(" + r + "," + t + ")"),
                ("(t,v)=(" + t + "," + v + ")")
        );
    }

    private Optional<V> findShortcut(V p, V v, GraphPath<V, E> tour, Set<V> used) {
        Graph<V, E> graph = tour.getGraph();
        E p_v = graph.getEdge(p, v);
        V preP = Graphs.getOppositeVertex(graph,
                tour.getEdgeList().stream()
                        .filter(e -> graph.edgesOf(p).contains(e) && !e.equals(p_v))
                        .findFirst().orElseThrow(() -> new RuntimeException("Could not find predecessor of p: " + p)),
                p);
        Set<V> toCheck = new HashSet<>(graph.vertexSet());
        toCheck.removeAll(used);
        toCheck.remove(p);
        toCheck.remove(preP); //do not shortcut to immediate neighbour in tour

        double curMinWeight = graph.getEdgeWeight(p_v);
        V r = null;

        for (V r_test : toCheck) {
            E p_r_test = graph.getEdge(p, r_test);
            double p_r_test_weight = graph.getEdgeWeight(p_r_test);
            if (p_r_test_weight < curMinWeight) {
                curMinWeight = p_r_test_weight;
                r = r_test;
            }
        }

        if (r == null) {
            return Optional.empty();
        }

        return Optional.of(r);
    }

    private GraphWalk<V, E> constructDeltaPath(GraphPath<V, E> tour, V p, V v, V r) {
        Graph<V, E> graph = tour.getGraph();
        E p_v = graph.getEdge(p, v);
        E p_r = graph.getEdge(p, r);

        List<E> edgeListReplaced = replaceInEdgeList(tour.getEdgeList(), p_v, p_r);
        GraphWalk<V, E> deltaPath = new GraphWalk<>(graph, v, r, edgeListReplaced,
                tour.getWeight() - graph.getEdgeWeight(p_v) + graph.getEdgeWeight(p_r)
        );

        assert tour.getEdgeList().size() == deltaPath.getEdgeList().size();

        return deltaPath;
    }

    private Pair<GraphPath<V, E>, V> transformDeltaPathToTour(GraphWalk<V, E> deltaPath, V p, V v, V r) {
        Graph<V, E> graph = deltaPath.getGraph();

        Graph<V, E> deltaPathAsGraph = new MaskSubgraph<>(deltaPath.getGraph(), vertex -> false, e -> !deltaPath.getEdgeList().contains(e));
        Set<V> cycle = new CycleDetector<>(
                directedUnweightedGraphFromEdgeList(deltaPathAsGraph, v, r)
        ).findCyclesContainingVertex(r); //problem -> directed (both edges given) -> can always find cycle over all vertices

        E r_t = deltaPathAsGraph.edgesOf(r).stream()
                .filter(e -> {
                    V opposite = Graphs.getOppositeVertex(graph, e, r);
                    return cycle.contains(opposite) && !opposite.equals(p);
                }) //f=(r,t), t!=p
                .findFirst()
                .orElseThrow(() -> new RuntimeException("Could not find edge f=(r,t) (which is to be deleted) to turn deltapath into a tour."));

        V t = Graphs.getOppositeVertex(graph, r_t, r);
        E t_v = graph.getEdge(t, v);

        Set<E> newEdgeSet = new HashSet<>(replaceInEdgeList(deltaPath.getEdgeList(), r_t, t_v));

        return new Pair<>(edgeSetToTour(newEdgeSet, graph), t);
    }

    private List<E> replaceInEdgeList(List<E> edges, E toReplace, E replaceWith) {
        if (!edges.contains(toReplace))
            throw new RuntimeException(String.format("Cannot replace edge %s, since it does not exist.", toReplace));
        return edges.stream()
                .map(e -> e.equals(toReplace) ? replaceWith : e)
                .toList();
    }

    private Graph<V, E> directedUnweightedGraphFromEdgeList(Graph<V, E> undirectedDeltaPathAsGraph, V v, V r) {
        Graph<V, E> directed = new SimpleDirectedGraph<>(undirectedDeltaPathAsGraph.getVertexSupplier(), undirectedDeltaPathAsGraph.getEdgeSupplier(), false);
        undirectedDeltaPathAsGraph.vertexSet().forEach(directed::addVertex);

        DepthFirstIterator<V, E> iterator = new DepthFirstIterator<>(undirectedDeltaPathAsGraph, v);
        V last = v;
        while (iterator.hasNext()) {
            v = iterator.next();
            if (last.equals(v)) continue;
            directed.addEdge(last, v);
            last = v;
        }
        if (directed.edgeSet().size() == undirectedDeltaPathAsGraph.edgeSet().size() - 1) directed.addEdge(last, r);
        return directed;
    }

}
